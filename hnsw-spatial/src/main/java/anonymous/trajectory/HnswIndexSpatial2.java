package anonymous.trajectory;

import com.github.jelmerk.knn.DistanceFunction;
import com.github.jelmerk.knn.Index;
import com.github.jelmerk.knn.Item;
import com.github.jelmerk.knn.JavaObjectSerializer;
import com.github.jelmerk.knn.ObjectSerializer;
import com.github.jelmerk.knn.ProgressListener;
import com.github.jelmerk.knn.SearchResult;
import com.github.jelmerk.knn.hnsw.SizeLimitExceededException;
import com.github.jelmerk.knn.util.ArrayBitSet;
import com.github.jelmerk.knn.util.BitSet;
import com.github.jelmerk.knn.util.ClassLoaderObjectInputStream;
import com.github.jelmerk.knn.util.GenericObjectPool;
import com.github.jelmerk.knn.util.SynchronizedBitSet;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.PriorityQueue;
import java.util.concurrent.atomic.AtomicReferenceArray;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.function.Predicate;

import org.eclipse.collections.api.list.MutableList;
import org.eclipse.collections.api.map.primitive.MutableObjectIntMap;
import org.eclipse.collections.api.map.primitive.MutableObjectLongMap;
import org.eclipse.collections.impl.list.mutable.FastList;
import org.eclipse.collections.impl.map.mutable.primitive.ObjectIntHashMap;
import org.eclipse.collections.impl.map.mutable.primitive.ObjectLongHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Implementation of {@link Index} that implements the hnsw algorithm (triangle inequality optimization disabled)
 *
 * @param <TId>       Type of the external identifier of an item
 * @param <TVector>   Type of the vector to perform distance calculation on
 * @param <TItem>     Type of items stored in the index
 * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
 * @see <a href="https://arxiv.org/abs/1603.09320">
 * Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs</a>
 */
public class HnswIndexSpatial2<TId, TVector, TItem extends Item<TId, TVector>, TDistance> implements
    Index<TId, TVector, TItem, TDistance> {

  final Logger logger = LoggerFactory.getLogger(HnswIndexSpatial2.class);

  private static final byte VERSION_1 = 0x01;

  private static final long serialVersionUID = 1L;

  private static final int NO_NODE_ID = -1;

  private DistanceFunction<TVector, TDistance> distanceFunction;
  private DistanceFunction<TVector, TDistance> constrDistanceFunction;
  private BinaryOperator<TDistance> differenceCalculator;
  private Comparator<TDistance> distanceComparator;
  private MaxValueComparator<TDistance> maxValueDistanceComparator;

  private int dimensions;
  private int maxItemCount;

  /**
   * num of neighbours added as links
   */
  private int m;

  /**
   * max num of links a vector can have
   */
  private int maxM;

  /**
   * max num of links a vector can have at layer 0
   */
  private int maxM0;
  private double levelLambda;
  private int ef;
  private int efConstruction;
  private boolean removeEnabled;

  private final int maxLevel;
  private Predicate<TItem> entryPointPicker;
  private Function<TVector, TItem> entryPointGetter;
  private Function<TVector, TItem> constrEntryPointGetter;

  private int nodeCount;

  private volatile Node<TItem, TDistance> fallbackEntryPoint;
  private final MutableList<Node<TItem, TDistance>> entryPoints;

  private AtomicReferenceArray<Node<TItem, TDistance>> nodes;

  /**
   * Find node's id by item's id
   */
  private MutableObjectIntMap<TId> lookup;

  private final MutableObjectIntMap<TId> entryPointLookup;

  private MutableObjectLongMap<TId> deletedItemVersions;
  private Map<TId, Object> locks;

  private ObjectSerializer<TId> itemIdSerializer;
  private ObjectSerializer<TItem> itemSerializer;

  private ReentrantLock globalLock;

  private GenericObjectPool<BitSet> visitedBitSetPool;

  private BitSet excludedCandidates;

  private ExactView exactView;

  private HnswIndexSpatial2(RefinedBuilder<TId, TVector, TItem, TDistance> builder) {

    this.dimensions = builder.dimensions;
    this.maxItemCount = builder.maxItemCount;
    this.distanceFunction = builder.distanceFunction;
    this.constrDistanceFunction = builder.constrDistanceFunction;
    this.distanceComparator = builder.distanceComparator;
    this.differenceCalculator = builder.differenceCalculator;
    this.maxValueDistanceComparator = new MaxValueComparator<>(this.distanceComparator);

    this.m = builder.m;
    this.maxM = builder.m;
    this.maxM0 = builder.m * 2;
    this.levelLambda = 1 / Math.log(this.m);  // m_L
    this.efConstruction = Math.max(builder.efConstruction, m);
    this.ef = builder.ef;
    this.removeEnabled = builder.removeEnabled;
    this.maxLevel = builder.maxLevel;

    this.nodes = new AtomicReferenceArray<>(this.maxItemCount);

    this.lookup = new ObjectIntHashMap<>();
    this.entryPointLookup = new ObjectIntHashMap<>();
    this.deletedItemVersions = new ObjectLongHashMap<>();
    this.locks = new HashMap<>();

    this.itemIdSerializer = builder.itemIdSerializer;
    this.itemSerializer = builder.itemSerializer;
    this.entryPointPicker = builder.entryPointPicker;
    this.entryPointGetter = builder.entryPointGetter;
    this.constrEntryPointGetter = builder.constrEntryPointGetter;

    this.entryPoints = FastList.newList();

    this.globalLock = new ReentrantLock();

    this.visitedBitSetPool = new GenericObjectPool<>(() -> new ArrayBitSet(this.maxItemCount),
                                                     Runtime.getRuntime().availableProcessors());

    this.excludedCandidates = new SynchronizedBitSet(new ArrayBitSet(this.maxItemCount));

    this.exactView = new ExactView();
  }

  /**
   * Restores a {@link HnswIndexSpatial2} from a File.
   *
   * @param file        File to restore the index from
   * @param <TId>       Type of the external identifier of an item
   * @param <TVector>   Type of the vector to perform distance calculation on
   * @param <TItem>     Type of items stored in the index
   * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
   * @return The restored index
   * @throws IOException in case of an I/O exception
   */
  public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance> HnswIndexSpatial2<TId, TVector, TItem, TDistance> load(
      File file) throws IOException {
    return load(new FileInputStream(file));
  }

  /**
   * Restores a {@link HnswIndexSpatial2} from a File.
   *
   * @param file        File to restore the index from
   * @param classLoader the classloader to use
   * @param <TId>       Type of the external identifier of an item
   * @param <TVector>   Type of the vector to perform distance calculation on
   * @param <TItem>     Type of items stored in the index
   * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
   * @return The restored index
   * @throws IOException in case of an I/O exception
   */
  public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance> HnswIndexSpatial2<TId, TVector, TItem, TDistance> load(
      File file, ClassLoader classLoader) throws IOException {
    return load(new FileInputStream(file), classLoader);
  }

  /**
   * Restores a {@link HnswIndexSpatial2} from a Path.
   *
   * @param path        Path to restore the index from
   * @param <TId>       Type of the external identifier of an item
   * @param <TVector>   Type of the vector to perform distance calculation on
   * @param <TItem>     Type of items stored in the index
   * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
   * @return The restored index
   * @throws IOException in case of an I/O exception
   */
  public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance> HnswIndexSpatial2<TId, TVector, TItem, TDistance> load(
      Path path) throws IOException {
    return load(Files.newInputStream(path));
  }

  /**
   * Restores a {@link HnswIndexSpatial2} from a Path.
   *
   * @param path        Path to restore the index from
   * @param classLoader the classloader to use
   * @param <TId>       Type of the external identifier of an item
   * @param <TVector>   Type of the vector to perform distance calculation on
   * @param <TItem>     Type of items stored in the index
   * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
   * @return The restored index
   * @throws IOException in case of an I/O exception
   */
  public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance> HnswIndexSpatial2<TId, TVector, TItem, TDistance> load(
      Path path, ClassLoader classLoader) throws IOException {
    return load(Files.newInputStream(path), classLoader);
  }

  /**
   * Restores a {@link HnswIndexSpatial2} from an InputStream.
   *
   * @param inputStream InputStream to restore the index from
   * @param <TId>       Type of the external identifier of an item
   * @param <TVector>   Type of the vector to perform distance calculation on
   * @param <TItem>     Type of items stored in the index
   * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ...).
   * @return The restored index
   * @throws IOException              in case of an I/O exception
   * @throws IllegalArgumentException in case the file cannot be read
   */
  public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance> HnswIndexSpatial2<TId, TVector, TItem, TDistance> load(
      InputStream inputStream) throws IOException {
    return load(inputStream, Thread.currentThread().getContextClassLoader());
  }

  /**
   * Restores a {@link HnswIndexSpatial2} from an InputStream.
   *
   * @param inputStream InputStream to restore the index from
   * @param classLoader the classloader to use
   * @param <TId>       Type of the external identifier of an item
   * @param <TVector>   Type of the vector to perform distance calculation on
   * @param <TItem>     Type of items stored in the index
   * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ...).
   * @return The restored index
   * @throws IOException              in case of an I/O exception
   * @throws IllegalArgumentException in case the file cannot be read
   */
  @SuppressWarnings("unchecked")
  public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance> HnswIndexSpatial2<TId, TVector, TItem, TDistance> load(
      InputStream inputStream, ClassLoader classLoader) throws IOException {

    try (ObjectInputStream ois = new ClassLoaderObjectInputStream(classLoader, inputStream)) {
      return (HnswIndexSpatial2<TId, TVector, TItem, TDistance>) ois.readObject();
    } catch (ClassNotFoundException e) {
      throw new IllegalArgumentException("Could not read input file.", e);
    }
  }


  /**
   * Start the process of building a new HNSW index.
   *
   * @param dimensions       the dimensionality of the vectors stored in the index
   * @param distanceFunction the distance function
   * @param maxItemCount     maximum number of items the index can hold
   * @param <TVector>        Type of the vector to perform distance calculation on
   * @param <TDistance>      Type of distance between items (expect any numeric type: float, double, int, ..)
   * @return a builder
   */
  public static <TVector, TDistance extends Comparable<TDistance>> Builder<TVector, TDistance> newBuilder(
      int dimensions, DistanceFunction<TVector, TDistance> distanceFunction,
      DistanceFunction<TVector, TDistance> constrDistanceFunction, BinaryOperator<TDistance> differenceCalculator,
      int maxItemCount) {

    Comparator<TDistance> distanceComparator = Comparator.naturalOrder();

    return new Builder<>(dimensions, distanceFunction, constrDistanceFunction, distanceComparator, differenceCalculator,
                         maxItemCount);
  }

  /**
   * Start the process of building a new HNSW index.
   *
   * @param dimensions         the dimensionality of the vectors stored in the index
   * @param distanceFunction   the distance function
   * @param distanceComparator used to compare distances
   * @param maxItemCount       maximum number of items the index can hold
   * @param <TVector>          Type of the vector to perform distance calculation on
   * @param <TDistance>        Type of distance between items (expect any numeric type: float, double, int, ..)
   * @return a builder
   */
  public static <TVector, TDistance> Builder<TVector, TDistance> newBuilder(
      int dimensions,
      DistanceFunction<TVector, TDistance> distanceFunction,
      DistanceFunction<TVector, TDistance> constrDistanceFunction,
      Comparator<TDistance> distanceComparator,
      BinaryOperator<TDistance> differenceCalculator,
      int maxItemCount) {

    return new Builder<>(dimensions, distanceFunction, constrDistanceFunction, distanceComparator, differenceCalculator,
                         maxItemCount);
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public int size() {
    globalLock.lock();
    try {
      return lookup.size();
    } finally {
      globalLock.unlock();
    }
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public Optional<TItem> get(TId id) {
    globalLock.lock();
    try {
      int nodeId = lookup.getIfAbsent(id, NO_NODE_ID);

      if (nodeId == NO_NODE_ID) {
        return Optional.empty();
      } else {
        return Optional.of(nodes.get(nodeId).item);
      }
    } finally {
      globalLock.unlock();
    }
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public Collection<TItem> items() {
    globalLock.lock();
    try {
      List<TItem> results = new ArrayList<>(size());

      Iterator<TItem> iter = new ItemIterator();

      while (iter.hasNext()) {
        results.add(iter.next());
      }

      return results;
    } finally {
      globalLock.unlock();
    }
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public boolean remove(TId id, long version) {

    if (!removeEnabled) {
      return false;
    }

    globalLock.lock();

    try {
      int internalNodeId = lookup.getIfAbsent(id, NO_NODE_ID);

      if (internalNodeId == NO_NODE_ID) {
        return false;
      }

      Node<TItem, TDistance> node = nodes.get(internalNodeId);

      if (version < node.item.version()) {
        return false;
      }

      node.deleted = true;

      lookup.remove(id);

      deletedItemVersions.put(id, version);

      return true;
    } finally {
      globalLock.unlock();
    }
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public boolean add(TItem item) {
    if (item.dimensions() != dimensions) {
      throw new IllegalArgumentException("Item does not have dimensionality of : " + dimensions);
    }

    // The level where item is inserted
    int randomLevel;

    boolean isEntryPoint = entryPointPicker.test(item);
    if (isEntryPoint) {
      randomLevel = maxLevel;
      logger.debug("Item {} is inserted at layer {} as an entry point", item.id(), randomLevel);
    } else {
      do {
        randomLevel = assignLevel(item.id(), this.levelLambda);
      } while (randomLevel > maxLevel);
      logger.debug("Item {} is inserted at layer {}", item.id(), randomLevel);
    }

    // the item's friend lists on each level
    FastList<FastList<NodeIdAndDistance<TDistance>>> connections = FastList.newList(randomLevel + 1);

    for (int level = 0; level <= randomLevel; level++) {
      int levelM = randomLevel == 0 ? maxM0 : maxM;
      connections.add(FastList.newList(levelM));
    }

    globalLock.lock();

    try {
      int existingNodeId = lookup.getIfAbsent(item.id(), NO_NODE_ID);

      if (existingNodeId != NO_NODE_ID) {
        // item with the same identifier already exists in the index
        if (!removeEnabled) {
          return false;
        }

        Node<TItem, TDistance> node = nodes.get(existingNodeId);
        if (item.version() < node.item.version()) {
          return false;
        }

        if (Objects.deepEquals(node.item.vector(), item.vector())) {
          // New version, same vector
          node.item = item;
          return true;
        } else {
          // New version, different vector -- delete the old
          remove(item.id(), item.version());
        }
      } else if (item.version() < deletedItemVersions.getIfAbsent(item.id(), -1)) {
        // Same or higher version has been deleted
        return false;
      }

      // The item doesn't exist, and no item with the same identifier and same or higher
      // version has been deleted before -- add it to the index
      if (nodeCount >= this.maxItemCount) {
        throw new SizeLimitExceededException("The number of elements exceeds the specified limit.");
      }

      int newNodeId = nodeCount++;

      // FIXME what is this?
      excludedCandidates.add(newNodeId);

      Node<TItem, TDistance> newNode = new Node<>(newNodeId, connections, item, false);

      nodes.set(newNodeId, newNode);
      lookup.put(item.id(), newNodeId);
      deletedItemVersions.remove(item.id());

      Object lock = locks.computeIfAbsent(item.id(), k -> new Object());

      Node<TItem, TDistance> entryPointCopy = fallbackEntryPoint;
      logger.debug("The fallback entry point for inserting {} is {}", item.id(),
                   (entryPointCopy != null) ? entryPointCopy.item.id() : "null");

      TItem entryPointItem = constrEntryPointGetter.apply(item.vector());
      if (entryPointItem != null) {
        int entryPointId = entryPointLookup.getIfAbsent(entryPointItem.id(), NO_NODE_ID);
        if (entryPointId != NO_NODE_ID) {
          entryPointCopy = entryPoints.get(entryPointId);
          logger.debug("The entry point for inserting {} is reassigned to {}", item.id(), entryPointCopy.item.id());
        }
      }

      // Several reasons why entryPointCopy may fall back to the original HNSW implementation:
      // 1. constrEntryPointGetter returns null.
      // 2. constrEntryPointGetter returns an item that is not assigned as an entry point.
      // Note that the fallback value is null before the first insertion.

      try {
        synchronized (lock) {
          synchronized (newNode) {
            if (!isEntryPoint && fallbackEntryPoint != null && randomLevel <= fallbackEntryPoint.maxLevel()) {
              // The entry point list and the fallback entry point will not change. They are safe for other threads to
              // use, so the lock can be released now.
              globalLock.unlock();
            }

            Node<TItem, TDistance> currObj = entryPointCopy;

            if (currObj != null) {
              if (newNode.maxLevel() < entryPointCopy.maxLevel()) {
                // The distance between the current point and the inserted point
                TDistance curDist = constrDistanceFunction.distance(item.vector(), currObj.item.vector());

                // Stage 1: Above the insertion layer
                for (int activeLevel = entryPointCopy.maxLevel(); activeLevel > newNode.maxLevel(); activeLevel--) {
                  boolean changed = true;
                  while (changed) {
                    changed = false;
                    synchronized (currObj) {
                      FastList<NodeIdAndDistance<TDistance>> candidateConnections = currObj.connections.get(
                          activeLevel);
                      for (NodeIdAndDistance<TDistance> candidateConnection : candidateConnections) {
                        int candidateId = candidateConnection.nodeId;
                        Node<TItem, TDistance> candidateNode = nodes.get(candidateId);
                        TDistance candidateDistance = constrDistanceFunction.distance(item.vector(),
                                                                                      candidateNode.item.vector());
                        if (lt(candidateDistance, curDist)) {
                          // Select nearest neighbor
                          curDist = candidateDistance;
                          currObj = candidateNode;
                          changed = true;
                        }
                      }
                    }
                  }
                }
              }

              // Stage 2: On and below insertion layer
              for (int level = Math.min(randomLevel, entryPointCopy.maxLevel()); level >= 0; level--) {
                // topCandidates contains efConstruction points closest to inserted item
                // sorted in descending order of distance to inserted item
                PriorityQueue<NodeIdAndDistance<TDistance>> topCandidates = searchBaseLayer(currObj,
                                                                                            item.vector(),
                                                                                            efConstruction, level,
                                                                                            true);
                if (entryPointCopy.deleted) {
                  // FIXME why? If entry point is deleted, add it to the candidates?
                  TDistance distance = constrDistanceFunction.distance(item.vector(),
                                                                       entryPointCopy.item.vector());
                  topCandidates.add(new NodeIdAndDistance<>(entryPointCopy.id, distance,
                                                            maxValueDistanceComparator));

                  if (topCandidates.size() > efConstruction) {
                    topCandidates.poll();
                  }
                }
                mutuallyConnectNewElement(newNode, topCandidates, level);
              }
            }

            // Now that the item is fully inserted, if it is assigned as an entry point, update the entry
            // point list
            if (isEntryPoint) {
              entryPoints.add(newNode);
              entryPointLookup.put(item.id(), entryPoints.size() - 1);
            }

            // zoom out to the highest level
            if (fallbackEntryPoint == null || newNode.maxLevel() > fallbackEntryPoint.maxLevel()) {
              // this is thread safe because we get the global lock when we add a level
              fallbackEntryPoint = newNode;
            }

            return true;
          }
        }
      } finally {
        excludedCandidates.remove(newNodeId);
      }
    } finally {
      if (globalLock.isHeldByCurrentThread()) {
        globalLock.unlock();
      }
    }
  }

  private void mutuallyConnectNewElement(Node<TItem, TDistance> newNode,
                                         PriorityQueue<NodeIdAndDistance<TDistance>> topCandidates, int level) {

    int bestN = level == 0 ? this.maxM0 : this.maxM;

    int newNodeId = newNode.id;
    TVector newItemVector = newNode.item.vector();
    FastList<NodeIdAndDistance<TDistance>> newItemConnections = newNode.connections.get(level);

    getNeighborsByHeuristic2(topCandidates, m);

    while (!topCandidates.isEmpty()) {
      // the distance from the candidate to the inserted point
      NodeIdAndDistance<TDistance> selectedNeighbour = topCandidates.poll();
      int selectedNeighbourId = selectedNeighbour.nodeId;

      if (excludedCandidates.contains(selectedNeighbourId)) {
        // Skip the nodes currently being added
        continue;
      }

      // construction distances are stored in the connections array
      newItemConnections.add(selectedNeighbour);

      Node<TItem, TDistance> neighbourNode = nodes.get(selectedNeighbourId);

      // Add reverse edge
      synchronized (neighbourNode) {

        TVector neighbourVector = neighbourNode.item.vector();

        FastList<NodeIdAndDistance<TDistance>> neighbourConnectionsAtLevel = neighbourNode.connections.get(level);

        if (neighbourConnectionsAtLevel.size() < bestN) {
          neighbourConnectionsAtLevel.add(
              new NodeIdAndDistance<>(newNodeId, selectedNeighbour.distance, maxValueDistanceComparator));
        } else {
          // finding the "weakest" element to replace it with the new one

          TDistance dMax = constrDistanceFunction.distance(newItemVector, neighbourNode.item.vector());

          Comparator<NodeIdAndDistance<TDistance>> comparator = Comparator.<NodeIdAndDistance<TDistance>>naturalOrder()
              .reversed();

          PriorityQueue<NodeIdAndDistance<TDistance>> candidates = new PriorityQueue<>(comparator);
          candidates.add(new NodeIdAndDistance<>(newNodeId, dMax, maxValueDistanceComparator));

          neighbourConnectionsAtLevel.forEach(pair -> {
            int id = pair.nodeId;
            TDistance dist = constrDistanceFunction.distance(neighbourVector, nodes.get(id).item.vector());
            candidates.add(new NodeIdAndDistance<>(id, dist, maxValueDistanceComparator));
          });

          getNeighborsByHeuristic2(candidates, bestN);

          neighbourConnectionsAtLevel.clear();

          while (!candidates.isEmpty()) {
            neighbourConnectionsAtLevel.add(candidates.poll());
          }
        }
      }
    }
  }

  private void getNeighborsByHeuristic2(PriorityQueue<NodeIdAndDistance<TDistance>> topCandidates, int m) {

    if (topCandidates.size() < m) {
      return;
    }

    PriorityQueue<NodeIdAndDistance<TDistance>> queueClosest = new PriorityQueue<>();
    List<NodeIdAndDistance<TDistance>> returnList = new ArrayList<>();

    while (!topCandidates.isEmpty()) {
      queueClosest.add(topCandidates.poll());
    }

    // Now topCandidates is empty, the nodes in queueClosest in reverse order

    while (!queueClosest.isEmpty()) {
      if (returnList.size() >= m) {
        break;
      }

      NodeIdAndDistance<TDistance> currentPair = queueClosest.poll();

      TDistance distToQuery = currentPair.distance;

      boolean good = true;
      for (NodeIdAndDistance<TDistance> secondPair : returnList) {

        TDistance curDist = constrDistanceFunction.distance(nodes.get(secondPair.nodeId).item.vector(),
                                                            nodes.get(currentPair.nodeId).item.vector());

        if (lt(curDist, distToQuery)) {
          // There exists a node in returnList whose distance to the current node
          // is smaller than the distance between the current node and the new node.
          // The current node is excluded, because we would like our neighbours to
          // be as expansive across the graph as possible.
          good = false;
          break;
        }

      }
      if (good) {
        returnList.add(currentPair);
      }
    }

    topCandidates.addAll(returnList);
  }

  /**
   * {@inheritDoc}
   */
  @Override
  public List<SearchResult<TItem, TDistance>> findNearest(TVector destination, int k) {

    if (fallbackEntryPoint == null) {
      return Collections.emptyList();
    }

    Node<TItem, TDistance> entryPointCopy = fallbackEntryPoint;

    logger.debug("The fallback entry point for searching {} is {}", destination, entryPointCopy.item.id());
    TItem entryPointItem = entryPointGetter.apply(destination);
    if (entryPointItem != null) {
      int entryPointId = entryPointLookup.getIfAbsent(entryPointItem.id(), NO_NODE_ID);
      if (entryPointId != NO_NODE_ID) {
        entryPointCopy = entryPoints.get(entryPointId);
        logger.debug("The entry point for searching {} is reassigned as {}", destination, entryPointCopy.item.id());
      }
    }

    // Several reasons why entryPointCopy may fall back to the original HNSW implementation:
    // 1. entryPointGetter returns null.
    // 2. entryPointGetter returns an item that is not assigned as an entry point.

    Node<TItem, TDistance> currObj = entryPointCopy;

    TDistance curDist = distanceFunction.distance(destination, currObj.item.vector());

    for (int activeLevel = entryPointCopy.maxLevel(); activeLevel > 0; activeLevel--) {

      boolean changed = true;

      while (changed) {
        changed = false;

        synchronized (currObj) {
          List<NodeIdAndDistance<TDistance>> candidateConnections = currObj.connections.get(activeLevel);

          for (NodeIdAndDistance<TDistance> candidateConnection : candidateConnections) {

            int candidateId = candidateConnection.nodeId;

            TDistance candidateDistance = distanceFunction.distance(destination,
                                                                    nodes.get(candidateId).item.vector());
            if (lt(candidateDistance, curDist)) {
              curDist = candidateDistance;
              currObj = nodes.get(candidateId);
              changed = true;
            }
          }
        }

      }
    }

    PriorityQueue<NodeIdAndDistance<TDistance>> topCandidates = searchBaseLayer(currObj, destination,
                                                                                Math.max(ef, k), 0, false);

    while (topCandidates.size() > k) {
      topCandidates.poll();
    }

    List<SearchResult<TItem, TDistance>> results = new ArrayList<>(topCandidates.size());
    while (!topCandidates.isEmpty()) {
      NodeIdAndDistance<TDistance> pair = topCandidates.poll();
      results.add(0, new SearchResult<>(nodes.get(pair.nodeId).item, pair.distance, maxValueDistanceComparator));
    }

    return results;
  }

  /**
   * Find k closest points to the destination on the specific layer. If {@code isConstruction} is set, construction
   * distance is returned, otherwise search distance.
   *
   * @param entryPointNode the point from which the search starts
   * @param destination    the destination point
   * @param k              number of results
   * @param layer          the specific layer
   * @param isConstruction if this function is used during construction
   * @return k closest points to the destination on the specific layer, sorted in descending order of distance
   */
  private PriorityQueue<NodeIdAndDistance<TDistance>> searchBaseLayer(Node<TItem, TDistance> entryPointNode,
                                                                      TVector destination,
                                                                      int k, int layer, boolean isConstruction) {

    DistanceFunction<TVector, TDistance> cdf = constrDistanceFunction;
    DistanceFunction<TVector, TDistance> df = isConstruction ? constrDistanceFunction : distanceFunction;

    BitSet visitedBitSet = visitedBitSetPool.borrowObject();

    try {
      // the result set; holds at most k elements; a subset of candidateSet
      PriorityQueue<NodeIdAndDistance<TDistance>> topCandidates = new PriorityQueue<>(
          Comparator.<NodeIdAndDistance<TDistance>>naturalOrder().reversed());
      // used for BFS
      PriorityQueue<NodeIdAndDistance<TDistance>> candidateSet = new PriorityQueue<>();

      TDistance lowerBound;

      if (!entryPointNode.deleted) {
        // Add the distance from entryPoint to destination first
        // The distance in the queue is the search distance
        TDistance distance = df.distance(destination, entryPointNode.item.vector());
        NodeIdAndDistance<TDistance> pair = new NodeIdAndDistance<>(entryPointNode.id, distance,
                                                                    maxValueDistanceComparator);

        topCandidates.add(pair);
        lowerBound = distance;
        candidateSet.add(pair);

      } else {
        lowerBound = MaxValueComparator.maxValue();
        NodeIdAndDistance<TDistance> pair = new NodeIdAndDistance<>(entryPointNode.id, lowerBound,
                                                                    maxValueDistanceComparator);
        candidateSet.add(pair);
      }

      // Mark entryPoint as visited
      visitedBitSet.add(entryPointNode.id);

      while (!candidateSet.isEmpty()) {
        NodeIdAndDistance<TDistance> currentPair = candidateSet.poll();

        if (gt(currentPair.distance, lowerBound)) {
          break;
        }

        Node<TItem, TDistance> node = nodes.get(currentPair.nodeId);
        // TODO: use constr distance or query distance?
        // TDistance nodeDistance = cdf.distance(destination, node.item.vector());
//        TDistance nodeDistance = currentPair.distance;
        // assert nodeDistance.equals(nodeDistance2);

        synchronized (node) {
          // Candidates: friends of `node` on this layer
          List<NodeIdAndDistance<TDistance>> candidates = node.connections.get(layer);

          for (NodeIdAndDistance<TDistance> candidate : candidates) {

            int candidateId = candidate.nodeId;

            if (!visitedBitSet.contains(candidateId)) {
              Node<TItem, TDistance> candidateNode = nodes.get(candidateId);

//              // Optimization by triangle inequality
//              TDistance difference = differenceCalculator.apply(nodeDistance, candidate.distance);
//
//              if (topCandidates.size() >= k) {
//                logger.debug("d1={}, b1={} (diff={}); cur_max={}",
//                             nodeDistance, candidate.distance, difference, lowerBound);
//                if (gt(difference, lowerBound)) {
//                  // TODO: why never entered?
//                  logger.debug("Optimized in {}: current item is {}, candidate item {} is not enqueued",
//                               isConstruction ? "construction" : "search", node.item.id(), candidateNode.item.id());
//                  continue;
//                }
//              }

              logger.debug("Current item is {}, candidate item {} is enqueued", node.item.id(),
                           candidateNode.item.id());

              visitedBitSet.add(candidateId);

              TDistance candidateDistance = df.distance(destination, candidateNode.item.vector());

              if (topCandidates.size() < k || gt(lowerBound, candidateDistance)) {
                // select the first k candidates closer to the destination
                NodeIdAndDistance<TDistance> candidatePair = new NodeIdAndDistance<>(candidateId,
                                                                                     candidateDistance,
                                                                                     maxValueDistanceComparator);

                candidateSet.add(candidatePair);

                if (!candidateNode.deleted) {
                  topCandidates.add(candidatePair);
                }

                if (topCandidates.size() > k) {
                  topCandidates.poll();
                }

                if (!topCandidates.isEmpty()) {
                  lowerBound = topCandidates.peek().distance;
                }
              }
            }
          }

        }
      }

      return topCandidates;
    } finally {
      visitedBitSet.clear();
      visitedBitSetPool.returnObject(visitedBitSet);
    }
  }

  /**
   * Creates a read only view on top of this index that uses pairwise comparison when doing distance search. And as such
   * can be used as a baseline for assessing the precision of the index. Searches will be really slow but give the
   * correct result every time.
   *
   * @return read only view on top of this index that uses pairwise comparison when doing distance search
   */
  public Index<TId, TVector, TItem, TDistance> asExactIndex() {
    return exactView;
  }

  /**
   * Returns the dimensionality of the items stored in this index.
   *
   * @return the dimensionality of the items stored in this index
   */
  public int getDimensions() {
    return dimensions;
  }

  /**
   * Returns the number of bidirectional links created for every new element during construction.
   *
   * @return the number of bidirectional links created for every new element during construction
   */
  public int getM() {
    return m;
  }

  /**
   * The size of the dynamic list for the nearest neighbors (used during the search)
   *
   * @return The size of the dynamic list for the nearest neighbors
   */
  public int getEf() {
    return ef;
  }

  /**
   * Set the size of the dynamic list for the nearest neighbors (used during the search)
   *
   * @param ef The size of the dynamic list for the nearest neighbors
   */
  public void setEf(int ef) {
    this.ef = ef;
  }

  /**
   * Returns the parameter has the same meaning as ef, but controls the index time / index precision.
   *
   * @return the parameter has the same meaning as ef, but controls the index time / index precision
   */
  public int getEfConstruction() {
    return efConstruction;
  }

  /**
   * Returns index of the highest layer in the graph. Entry points are inserted to this layer. Note that because of
   * layer 0, there are (1 + getMaxLevel()) layers in total.
   *
   * @return the max n
   */
  public int getMaxLevel() {
    return maxLevel;
  }

  /**
   * Returns the distance function for search.
   *
   * @return the distance function for search
   */
  public DistanceFunction<TVector, TDistance> getDistanceFunction() {
    return distanceFunction;
  }

  /**
   * Returns the distance function for construction.
   *
   * @return the distance function for construction
   */
  public DistanceFunction<TVector, TDistance> getConstrDistanceFunction() {
    return constrDistanceFunction;
  }

  /**
   * Returns the comparator used to compare distances.
   *
   * @return the comparator used to compare distance
   */
  public Comparator<TDistance> getDistanceComparator() {
    return distanceComparator;
  }


  /**
   * Returns if removes are enabled.
   *
   * @return true if removes are enabled for this index.
   */
  public boolean isRemoveEnabled() {
    return removeEnabled;
  }

  /**
   * Returns the maximum number of items the index can hold.
   *
   * @return the maximum number of items the index can hold
   */
  public int getMaxItemCount() {
    return maxItemCount;
  }

  /**
   * Returns the serializer used to serialize item id's when saving the index.
   *
   * @return the serializer used to serialize item id's when saving the index
   */
  public ObjectSerializer<TId> getItemIdSerializer() {
    return itemIdSerializer;
  }

  /**
   * Returns the serializer used to serialize items when saving the index.
   *
   * @return the serializer used to serialize items when saving the index
   */
  public ObjectSerializer<TItem> getItemSerializer() {
    return itemSerializer;
  }

  /**
   * Returns the entry point getter used for construction.
   *
   * @return the entry point getter used for construction
   */
  public Function<TVector, TItem> getConstrEntryPointGetter() {
    return constrEntryPointGetter;
  }

  /**
   * Returns the entry point getter used for queries.
   *
   * @return the entry point getter used for queries
   */
  public Function<TVector, TItem> getEntryPointGetter() {
    return entryPointGetter;
  }


  /**
   * The entry point getter used for construction
   *
   * @param constrEntryPointGetter the entry point getter used for construction
   */
  public void setConstrEntryPointGetter(Function<TVector, TItem> constrEntryPointGetter) {
    this.constrEntryPointGetter = constrEntryPointGetter;
  }

  /**
   * The entry point getter used for search
   *
   * @param entryPointGetter the entry point getter used for search
   */
  public void setEntryPointGetter(Function<TVector, TItem> entryPointGetter) {
    this.entryPointGetter = entryPointGetter;
  }

  /**
   * The function that decides if an inserted point is an entry point
   *
   * @param entryPointPicker the function that decides if an inserted point is an entry point
   */
  public void setEntryPointPicker(Predicate<TItem> entryPointPicker) {
    this.entryPointPicker = entryPointPicker;
  }

  /**
   * Checks if an item is an entry point of the index. An entry point is inserted at the highest level of the graph.
   *
   * @param id the id of the item
   * @return if the item is an entry point
   */
  public boolean isEntryPoint(TId id) {
    return entryPoints.stream().anyMatch((node) -> node.item.id() == id);
  }

  /**
   * Save (not supported)
   *
   * @param out the output stream to write the index to
   * @throws IOException
   */
  @Override
  public void save(OutputStream out) throws IOException {
    throw new UnsupportedOperationException("Saving is not supported.");
  }

//  /**
//   * {@inheritDoc}
//   */
//  @Override
//  public void save(OutputStream out) throws IOException {
//    try (ObjectOutputStream oos = new ObjectOutputStream(out)) {
//      oos.writeObject(this);
//    }
//  }
//
//  private void writeObject(ObjectOutputStream oos) throws IOException {
//    oos.writeByte(VERSION_1);
//    oos.writeInt(dimensions);
//    oos.writeObject(distanceFunction);
//    oos.writeObject(distanceComparator);
//    oos.writeObject(itemIdSerializer);
//    oos.writeObject(itemSerializer);
//    oos.writeInt(maxItemCount);
//    oos.writeInt(m);
//    oos.writeInt(maxM);
//    oos.writeInt(maxM0);
//    oos.writeDouble(levelLambda);
//    oos.writeInt(ef);
//    oos.writeInt(efConstruction);
//    oos.writeBoolean(removeEnabled);
//    oos.writeInt(nodeCount);
//    writeMutableObjectIntMap(oos, lookup);
//    writeMutableObjectLongMap(oos, deletedItemVersions);
//    writeNodesArray(oos, nodes);
//    oos.writeInt(fallbackEntryPoint == null ? -1 : fallbackEntryPoint.id);
//  }
//
//  @SuppressWarnings("unchecked")
//  private void readObject(ObjectInputStream ois) throws IOException, ClassNotFoundException {
//    @SuppressWarnings("unused") byte version = ois.readByte(); // for coping with future incompatible serialization
//    this.dimensions = ois.readInt();
//    this.distanceFunction = (DistanceFunction<TVector, TDistance>) ois.readObject();
//    this.distanceComparator = (Comparator<TDistance>) ois.readObject();
//    this.maxValueDistanceComparator = new MaxValueComparator<>(distanceComparator);
//    this.itemIdSerializer = (ObjectSerializer<TId>) ois.readObject();
//    this.itemSerializer = (ObjectSerializer<TItem>) ois.readObject();
//
//    this.maxItemCount = ois.readInt();
//    this.m = ois.readInt();
//    this.maxM = ois.readInt();
//    this.maxM0 = ois.readInt();
//    this.levelLambda = ois.readDouble();
//    this.ef = ois.readInt();
//    this.efConstruction = ois.readInt();
//    this.removeEnabled = ois.readBoolean();
//    this.nodeCount = ois.readInt();
//    this.lookup = readMutableObjectIntMap(ois, itemIdSerializer);
//    this.deletedItemVersions = readMutableObjectLongMap(ois, itemIdSerializer);
//    this.nodes = readNodesArray(ois, itemSerializer, maxM0, maxM);
//
//    int entrypointNodeId = ois.readInt();
//    this.fallbackEntryPoint = entrypointNodeId == -1 ? null : nodes.get(entrypointNodeId);
//
//    this.globalLock = new ReentrantLock();
//    this.visitedBitSetPool = new GenericObjectPool<>(() -> new ArrayBitSet(this.maxItemCount),
//                                                     Runtime.getRuntime().availableProcessors());
//    this.excludedCandidates = new SynchronizedBitSet(new ArrayBitSet(this.maxItemCount));
//    this.locks = new HashMap<>();
//    this.exactView = new ExactView();
//  }
//
//  private void writeMutableObjectIntMap(ObjectOutputStream oos, MutableObjectIntMap<TId> map) throws IOException {
//    oos.writeInt(map.size());
//
//    for (ObjectIntPair<TId> pair : map.keyValuesView()) {
//      itemIdSerializer.write(pair.getOne(), oos);
//      oos.writeInt(pair.getTwo());
//    }
//  }
//
//  private void writeMutableObjectLongMap(ObjectOutputStream oos, MutableObjectLongMap<TId> map) throws IOException {
//    oos.writeInt(map.size());
//
//    for (ObjectLongPair<TId> pair : map.keyValuesView()) {
//      itemIdSerializer.write(pair.getOne(), oos);
//      oos.writeLong(pair.getTwo());
//    }
//  }
//
//  private void writeNodesArray(ObjectOutputStream oos, AtomicReferenceArray<Node<TItem, TDistance>> nodes) throws IOException {
//    oos.writeInt(nodes.length());
//    for (int i = 0; i < nodes.length(); i++) {
//      writeNode(oos, nodes.get(i));
//    }
//  }
//
//  private void writeNode(ObjectOutputStream oos, Node<TItem, TDistance> node) throws IOException {
//    if (node == null) {
//      oos.writeInt(-1);
//    } else {
//      oos.writeInt(node.id);
//      oos.writeInt(node.connections.size());
//
//      for (MutableIntList connections : node.connections) {
//        oos.writeInt(connections.size());
//        for (int j = 0; j < connections.size(); j++) {
//          oos.writeInt(connections.get(j));
//        }
//      }
//      itemSerializer.write(node.item, oos);
//      oos.writeBoolean(node.deleted);
//    }
//  }

//  private static IntArrayList readIntArrayList(ObjectInputStream ois, int initialSize) throws IOException {
//    int size = ois.readInt();
//
//    IntArrayList list = new IntArrayList(initialSize);
//
//    for (int j = 0; j < size; j++) {
//      list.add(ois.readInt());
//    }
//
//    return list;
//  }
//
//  private static <TDistance> FastList<NodeIdAndDistance<TDistance>> readNodeIdAndDistanceList(ObjectInputStream ois, int initialSize) throws IOException {
//    int size = ois.readInt();
//
//    FastList<NodeIdAndDistance<TDistance>> list = FastList.newList(initialSize);
//
//    for (int j = 0; j < size; j++) {
//      list.add(new NodeIdAndDistance<TDistance>(ois.readInt(), ois.read    ))
//    }
//  }
//
//  private static <TItem, TDistance> Node<TItem, TDistance> readNode(ObjectInputStream ois,
//                                                                    ObjectSerializer<TItem> itemSerializer,
//                                                                    int maxM0, int maxM)
//      throws IOException, ClassNotFoundException {
//
//    int id = ois.readInt();
//
//    if (id == -1) {
//      return null;
//    } else {
//      int connectionsSize = ois.readInt();
//
//      FastList<FastList<NodeIdAndDistance<TDistance>>> connections = FastList.newList(connectionsSize);
//
//      for (int i = 0; i < connectionsSize; i++) {
//        int levelM = i == 0 ? maxM0 : maxM;
//        connections.set(i, readIntArrayList(ois, levelM));
//      }
//
//      TItem item = itemSerializer.read(ois);
//
//      boolean deleted = ois.readBoolean();
//
//      return new Node<>(id, connections, item, deleted);
//    }
//  }
//
//  private static <TItem, TDistance> AtomicReferenceArray<Node<TItem, TDistance>> readNodesArray(ObjectInputStream ois,
//                                                                                                ObjectSerializer<TItem> itemSerializer,
//                                                                                                int maxM0, int maxM)
//      throws IOException, ClassNotFoundException {
//
//    int size = ois.readInt();
//    AtomicReferenceArray<Node<TItem, TDistance>> nodes = new AtomicReferenceArray<>(size);
//
//    for (int i = 0; i < nodes.length(); i++) {
//      nodes.set(i, readNode(ois, itemSerializer, maxM0, maxM));
//    }
//
//    return nodes;
//  }
//
//  private static <TId> MutableObjectIntMap<TId> readMutableObjectIntMap(ObjectInputStream ois,
//                                                                        ObjectSerializer<TId> itemIdSerializer)
//      throws IOException, ClassNotFoundException {
//
//    int size = ois.readInt();
//
//    MutableObjectIntMap<TId> map = new ObjectIntHashMap<>(size);
//
//    for (int i = 0; i < size; i++) {
//      TId key = itemIdSerializer.read(ois);
//      int value = ois.readInt();
//
//      map.put(key, value);
//    }
//    return map;
//  }
//
//  private static <TId> MutableObjectLongMap<TId> readMutableObjectLongMap(ObjectInputStream ois,
//                                                                          ObjectSerializer<TId> itemIdSerializer)
//      throws IOException, ClassNotFoundException {
//
//    int size = ois.readInt();
//
//    MutableObjectLongMap<TId> map = new ObjectLongHashMap<>(size);
//
//    for (int i = 0; i < size; i++) {
//      TId key = itemIdSerializer.read(ois);
//      long value = ois.readLong();
//
//      map.put(key, value);
//    }
//    return map;
//  }

  private int assignLevel(TId value, double lambda) {

    // by relying on the external id to come up with the level, the graph construction should be a lot more stable
    // see : https://github.com/nmslib/hnswlib/issues/28

//    int hashCode = value.hashCode();
//    byte[] bytes = new byte[]{(byte) (hashCode >> 24), (byte) (hashCode >> 16), (byte) (hashCode >> 8),
//        (byte) hashCode};
//    double random = Math.abs((double) Murmur3.hash32(bytes) / (double) Integer.MAX_VALUE); // [0, 1]

    double random = Math.random();  // [0, 1]
    double r = -Math.log(random) * lambda;
    return (int) r;
  }

  private boolean lt(TDistance x, TDistance y) {
    return maxValueDistanceComparator.compare(x, y) < 0;
  }

  private boolean gt(TDistance x, TDistance y) {
    return maxValueDistanceComparator.compare(x, y) > 0;
  }

  static class Node<TItem, TDistance> implements Serializable {

    private static final long serialVersionUID = 1L;

    final int id;

    final FastList<FastList<NodeIdAndDistance<TDistance>>> connections;

    volatile TItem item;

    volatile boolean deleted;

    Node(int id, FastList<FastList<NodeIdAndDistance<TDistance>>> connections, TItem item, boolean deleted) {
      this.id = id;
      this.connections = connections;
      this.item = item;
      this.deleted = deleted;
    }

    int maxLevel() {
      return this.connections.size() - 1;
    }

    @Override
    public String toString() {
      final StringBuilder bd = new StringBuilder();
      bd.append("- id: ").append(id).append('\n');
      bd.append("  item: ").append(item).append('\n');
      bd.append("  connections:\n");
      for (int i = 0; i < connections.size(); i++) {
        bd.append("    layer ").append(i).append(": ");
        FastList<NodeIdAndDistance<TDistance>> connectionsInLayer = connections.get(i);
        for (NodeIdAndDistance<TDistance> connection : connectionsInLayer) {
          bd.append('(').append(connection.nodeId).append(", ").append(connection.distance).append(") ");
        }
        bd.append('\n');
      }
      return bd.toString();
    }
  }

  static class NodeIdAndDistance<TDistance> implements Comparable<NodeIdAndDistance<TDistance>> {

    final int nodeId;
    final TDistance distance;
    final Comparator<TDistance> distanceComparator;

    NodeIdAndDistance(int nodeId, TDistance distance, Comparator<TDistance> distanceComparator) {
      this.nodeId = nodeId;
      this.distance = distance;
      this.distanceComparator = distanceComparator;
    }

    @Override
    public int compareTo(NodeIdAndDistance<TDistance> o) {
      return distanceComparator.compare(distance, o.distance);
    }
  }

  static class MaxValueComparator<TDistance> implements Comparator<TDistance>, Serializable {

    private static final long serialVersionUID = 1L;

    private final Comparator<TDistance> delegate;

    MaxValueComparator(Comparator<TDistance> delegate) {
      this.delegate = delegate;
    }

    static <TDistance> TDistance maxValue() {
      return null;
    }

    @Override
    public int compare(TDistance o1, TDistance o2) {
      return o1 == null ? o2 == null ? 0 : 1 : o2 == null ? -1 : delegate.compare(o1, o2);
    }
  }

  /**
   * Base class for HNSW index builders.
   *
   * @param <TBuilder>  Concrete class that extends from this builder
   * @param <TVector>   Type of the vector to perform distance calculation on
   * @param <TDistance> Type of items stored in the index
   */
  public static abstract class BuilderBase<TBuilder extends BuilderBase<TBuilder, TVector, TDistance>, TVector, TDistance> {

    public static final int DEFAULT_M = 10;
    public static final int DEFAULT_EF = 10;
    public static final int DEFAULT_EF_CONSTRUCTION = 200;
    public static final boolean DEFAULT_REMOVE_ENABLED = false;

    int dimensions;
    DistanceFunction<TVector, TDistance> distanceFunction;
    DistanceFunction<TVector, TDistance> constrDistanceFunction;
    Comparator<TDistance> distanceComparator;
    BinaryOperator<TDistance> differenceCalculator;

    int maxItemCount;

    int m = DEFAULT_M;
    int ef = DEFAULT_EF;
    int efConstruction = DEFAULT_EF_CONSTRUCTION;
    boolean removeEnabled = DEFAULT_REMOVE_ENABLED;

    // The index of the highest layer in the HNSW graph. Entry points will be inserted to this layer.
    // Note that there are (1 + maxLevel) layers in total, including layer 0.
    int maxLevel = Integer.MAX_VALUE;

    BuilderBase(int dimensions, DistanceFunction<TVector, TDistance> distanceFunction,
                DistanceFunction<TVector, TDistance> constrDistanceFunction, Comparator<TDistance> distanceComparator,
                BinaryOperator<TDistance> differenceCalculator, int maxItemCount) {

      this.dimensions = dimensions;
      this.distanceFunction = distanceFunction;
      this.constrDistanceFunction = constrDistanceFunction;
      this.distanceComparator = distanceComparator;
      this.differenceCalculator = differenceCalculator;
      this.maxItemCount = maxItemCount;
    }

    abstract TBuilder self();

    /**
     * Sets the number of bidirectional links created for every new element during construction. Reasonable range for m
     * is 2-100. Higher m work better on datasets with high intrinsic dimensionality and/or high recall, while low m
     * work better for datasets with low intrinsic dimensionality and/or low recalls. The parameter also determines the
     * algorithm's memory consumption. As an example for d = 4 random vectors optimal m for search is somewhere around
     * 6, while for high dimensional datasets (word embeddings, good face descriptors), higher M are required (e.g. m =
     * 48, 64) for optimal performance at high recall. The range mM = 12-48 is OK for the most of the use cases. When m
     * is changed one has to update the other parameters. Nonetheless, ef and efConstruction parameters can be roughly
     * estimated by assuming that m  efConstruction is a constant.
     *
     * @param m the number of bidirectional links created for every new element during construction
     * @return the builder.
     */
    public TBuilder withM(int m) {
      this.m = m;
      return self();
    }

    /**
     * ` The parameter has the same meaning as ef, but controls the index time / index precision. Bigger efConstruction
     * leads to longer construction, but better index quality. At some point, increasing efConstruction does not improve
     * the quality of the index. One way to check if the selection of ef_construction was OK is to measure a recall for
     * M nearest neighbor search when ef = efConstruction: if the recall is lower than 0.9, then there is room for
     * improvement.
     *
     * @param efConstruction controls the index time / index precision
     * @return the builder
     */
    public TBuilder withEfConstruction(int efConstruction) {
      this.efConstruction = efConstruction;
      return self();
    }

    /**
     * The size of the dynamic list for the nearest neighbors (used during the search). Higher ef leads to more accurate
     * but slower search. The value ef of can be anything between k and the size of the dataset.
     *
     * @param ef size of the dynamic list for the nearest neighbors
     * @return the builder
     */
    public TBuilder withEf(int ef) {
      this.ef = ef;
      return self();
    }

    /**
     * Call to enable support for the experimental remove operation. Indices that support removes will consume more
     * memory.
     *
     * @return the builder
     */
    public TBuilder withRemoveEnabled() {
      this.removeEnabled = true;
      return self();
    }

    /**
     * Set the index of the highest layer in the graph. Entry points are inserted to this layer. Note that because of
     * layer 0, there will be (1 + maxLevel) layers in total.
     *
     * @param maxLevel the index of the highest layer
     * @return the builder
     */
    public TBuilder withMaxLevel(int maxLevel) {
      this.maxLevel = maxLevel;
      return self();
    }
  }

  /**
   * Builder for initializing an {@link HnswIndexSpatial2} instance.
   *
   * @param <TVector>   Type of the vector to perform distance calculation on
   * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
   */
  public static class Builder<TVector, TDistance> extends
      BuilderBase<Builder<TVector, TDistance>, TVector, TDistance> {

    /**
     * Constructs a new {@link Builder} instance.
     *
     * @param dimensions       the dimensionality of the vectors stored in the index
     * @param distanceFunction the distance function
     * @param maxItemCount     the maximum number of elements in the index
     */
    Builder(int dimensions, DistanceFunction<TVector, TDistance> distanceFunction,
            DistanceFunction<TVector, TDistance> constrDistanceFunction, Comparator<TDistance> distanceComparator,
            BinaryOperator<TDistance> differenceCalculator, int maxItemCount) {

      super(dimensions, distanceFunction, constrDistanceFunction, distanceComparator, differenceCalculator,
            maxItemCount);
    }

    @Override
    Builder<TVector, TDistance> self() {
      return this;
    }

    /**
     * Register the serializers used when saving the index.
     *
     * @param itemIdSerializer serializes the key of the item
     * @param itemSerializer   serializes the
     * @param <TId>            Type of the external identifier of an item
     * @param <TItem>          implementation of the Item interface
     * @return the builder
     */
    public <TId, TItem extends Item<TId, TVector>> RefinedBuilder<TId, TVector, TItem, TDistance> withCustomSerializers(
        ObjectSerializer<TId> itemIdSerializer, ObjectSerializer<TItem> itemSerializer) {
      return this.<TId, TItem>refine().withCustomSerializers(itemIdSerializer, itemSerializer);
    }

    /**
     * Register the entry point picker, which tells whether an item is inserted as an entry point.
     *
     * @param entryPointPicker the entry point picker
     * @return the builder
     */
    public <TId, TItem extends Item<TId, TVector>> RefinedBuilder<TId, TVector, TItem, TDistance> withEntryPointPicker(
        Predicate<TItem> entryPointPicker) {
      return this.<TId, TItem>refine().withEntryPointPicker(entryPointPicker);
    }

    /**
     * Register the entry point getters used upon construction and search.
     *
     * @param entryPointGetter       the entry point getter for search
     * @param constrEntryPointGetter the entry point getter for construction
     * @return the builder
     */
    public <TId, TItem extends Item<TId, TVector>> RefinedBuilder<TId, TVector, TItem, TDistance> withEntryPointGetters(
        Function<TVector, TItem> entryPointGetter, Function<TVector, TItem> constrEntryPointGetter) {
      return this.<TId, TItem>refine().withEntryPointGetters(entryPointGetter, constrEntryPointGetter);
    }

    /**
     * Build the index that uses java object serializers to store the items when reading and writing the index.
     *
     * @param <TId>   Type of the external identifier of an item
     * @param <TItem> implementation of the Item interface
     * @return the hnsw index instance
     */
    public <TId, TItem extends Item<TId, TVector>> HnswIndexSpatial2<TId, TVector, TItem, TDistance> build()
        throws IOException {
      return this.<TId, TItem>refine().build();
    }

    public <TId, TItem extends Item<TId, TVector>> RefinedBuilder<TId, TVector, TItem, TDistance> refine() {
      return new RefinedBuilder<>(dimensions, distanceFunction, constrDistanceFunction, distanceComparator,
                                  differenceCalculator, maxItemCount, m, ef,
                                  efConstruction, removeEnabled, maxLevel, null, null, null, null, null);
    }
  }

  /**
   * Extension of {@link Builder} that has knows what type of item is going to be stored in the index.
   *
   * @param <TId>       Type of the external identifier of an item
   * @param <TVector>   Type of the vector to perform distance calculation on
   * @param <TItem>     Type of items stored in the index
   * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
   */
  public static class RefinedBuilder<TId, TVector, TItem extends Item<TId, TVector>, TDistance> extends
      BuilderBase<RefinedBuilder<TId, TVector, TItem, TDistance>, TVector, TDistance> {

    private ObjectSerializer<TId> itemIdSerializer; // required
    private ObjectSerializer<TItem> itemSerializer; // required

    // Decide whether the point to be inserted is an entry point.
    private Predicate<TItem> entryPointPicker = null;
    // Decide the entry point of a query.
    private Function<TVector, TItem> entryPointGetter = null;
    // Decide the entry point when inserting a point during construction.
    private Function<TVector, TItem> constrEntryPointGetter = null;

    RefinedBuilder(
        int dimensions,
        DistanceFunction<TVector, TDistance> distanceFunction,
        DistanceFunction<TVector, TDistance> constrDistanceFunction,
        Comparator<TDistance> distanceComparator,
        BinaryOperator<TDistance> differenceCalculator,
        int maxItemCount,
        int m,
        int ef,
        int efConstruction,
        boolean removeEnabled,
        int maxLevel,
        ObjectSerializer<TId> itemIdSerializer,
        ObjectSerializer<TItem> itemSerializer,
        Predicate<TItem> entryPointPicker,
        Function<TVector, TItem> entryPointGetter,
        Function<TVector, TItem> constrEntryPointGetter
    ) {

      super(dimensions, distanceFunction, constrDistanceFunction, distanceComparator, differenceCalculator,
            maxItemCount);

      this.m = m;
      this.ef = ef;
      this.efConstruction = efConstruction;
      this.removeEnabled = removeEnabled;
      this.maxLevel = maxLevel;

      this.itemIdSerializer = itemIdSerializer;
      this.itemSerializer = itemSerializer;

      this.entryPointPicker = entryPointPicker;
      this.entryPointGetter = entryPointGetter;
      this.constrEntryPointGetter = constrEntryPointGetter;
    }

    @Override
    RefinedBuilder<TId, TVector, TItem, TDistance> self() {
      return this;
    }

    /**
     * Register the serializers used when saving the index.
     *
     * @param itemIdSerializer serializes the key of the item
     * @param itemSerializer   serializes the
     * @return the builder
     */
    public RefinedBuilder<TId, TVector, TItem, TDistance> withCustomSerializers(
        ObjectSerializer<TId> itemIdSerializer, ObjectSerializer<TItem> itemSerializer) {

      this.itemIdSerializer = itemIdSerializer;
      this.itemSerializer = itemSerializer;

      return this;
    }

    /**
     * Register the entry point getters used upon construction and search.
     * <p>
     * The entry point getters should return a {@code TItem} object which will be the entry point for the insertion or
     * search. If the getters return {@code null}, the entry point is chosen arbitrarily among the entry points
     * previously set (or falls back to the original HNSW implementation if none is set).
     *
     * @param entryPointGetter       the entry point getter for search
     * @param constrEntryPointGetter the entry point getter for construction
     * @return the builder
     */
    public RefinedBuilder<TId, TVector, TItem, TDistance> withEntryPointGetters(
        Function<TVector, TItem> entryPointGetter, Function<TVector, TItem> constrEntryPointGetter
    ) {
      this.entryPointGetter = entryPointGetter;
      this.constrEntryPointGetter = constrEntryPointGetter;

      return this;
    }

    /**
     * Register the entry point picker, which tells whether an item is inserted as an entry point.
     * <p>
     * The picker should return true if the item is an entry point, false otherwise. The entry point falls back to the
     * original HNSW implementation until the first entry point is set via the picker.
     *
     * @param entryPointPicker the entry point picker
     * @return the builder
     */
    public RefinedBuilder<TId, TVector, TItem, TDistance> withEntryPointPicker(Predicate<TItem> entryPointPicker) {
      this.entryPointPicker = entryPointPicker;
      return this;
    }

    /**
     * Build the index.
     *
     * @return the hnsw index instance
     */
    public HnswIndexSpatial2<TId, TVector, TItem, TDistance> build()  {
      // Assign default values
      if (this.itemIdSerializer == null) {
        this.itemIdSerializer = new JavaObjectSerializer<>();
      }
      if (this.itemSerializer == null) {
        this.itemSerializer = new JavaObjectSerializer<>();
      }
      if (this.entryPointPicker == null) {
        this.entryPointPicker = (item) -> false;
      }
      if (this.entryPointGetter == null) {
        this.entryPointGetter = (item) -> null;
      }
      if (this.constrEntryPointGetter == null) {
        this.constrEntryPointGetter = (item) -> null;
      }
      return new HnswIndexSpatial2<>(this);
    }


  }

  class ExactView implements Index<TId, TVector, TItem, TDistance> {

    private static final long serialVersionUID = 1L;

    @Override
    public int size() {
      return HnswIndexSpatial2.this.size();
    }

    @Override
    public Optional<TItem> get(TId tId) {
      return HnswIndexSpatial2.this.get(tId);
    }

    @Override
    public Collection<TItem> items() {
      return HnswIndexSpatial2.this.items();
    }

    @Override
    public List<SearchResult<TItem, TDistance>> findNearest(TVector vector, int k) {

      Comparator<SearchResult<TItem, TDistance>> comparator = Comparator.<SearchResult<TItem, TDistance>>naturalOrder()
          .reversed();

      PriorityQueue<SearchResult<TItem, TDistance>> queue = new PriorityQueue<>(k, comparator);

      for (int i = 0; i < nodeCount; i++) {
        Node<TItem, TDistance> node = nodes.get(i);
        if (node == null || node.deleted) {
          continue;
        }

        TDistance distance = distanceFunction.distance(node.item.vector(), vector);

        SearchResult<TItem, TDistance> searchResult = new SearchResult<>(node.item, distance,
                                                                         maxValueDistanceComparator);
        queue.add(searchResult);

        if (queue.size() > k) {
          queue.poll();
        }
      }

      List<SearchResult<TItem, TDistance>> results = new ArrayList<>(queue.size());

      SearchResult<TItem, TDistance> result;
      while ((result = queue.poll())
          != null) { // if you iterate over a priority queue the order is not guaranteed
        results.add(0, result);
      }

      return results;
    }

    @Override
    public boolean add(TItem item) {
      return HnswIndexSpatial2.this.add(item);
    }

    @Override
    public boolean remove(TId id, long version) {
      return HnswIndexSpatial2.this.remove(id, version);
    }

    @Override
    public void save(OutputStream out) throws IOException {
      HnswIndexSpatial2.this.save(out);
    }

    @Override
    public void save(File file) throws IOException {
      HnswIndexSpatial2.this.save(file);
    }

    @Override
    public void save(Path path) throws IOException {
      HnswIndexSpatial2.this.save(path);
    }

    @Override
    public void addAll(Collection<TItem> items) throws InterruptedException {
      HnswIndexSpatial2.this.addAll(items);
    }

    @Override
    public void addAll(Collection<TItem> items, ProgressListener listener) throws InterruptedException {
      HnswIndexSpatial2.this.addAll(items, listener);
    }

    @Override
    public void addAll(Collection<TItem> items, int numThreads, ProgressListener listener,
                       int progressUpdateInterval) throws InterruptedException {
      HnswIndexSpatial2.this.addAll(items, numThreads, listener, progressUpdateInterval);
    }
  }

  class ItemIterator implements Iterator<TItem> {

    private int done = 0;
    private int index = 0;

    @Override
    public boolean hasNext() {
      return done < HnswIndexSpatial2.this.size();
    }

    @Override
    public TItem next() {
      Node<TItem, TDistance> node;

      do {
        node = HnswIndexSpatial2.this.nodes.get(index++);
      } while (node == null || node.deleted);

      done++;

      return node.item;
    }
  }


  @Override
  public String toString() {
    final StringBuffer sb = new StringBuffer("HnswIndexSpatial:\n");
    sb.append("maxLevel: ").append(maxLevel).append('\n');
    sb.append("entryPoints: ");
    for (int i = 0; i < entryPoints.size(); i++) {
      Node<TItem, TDistance> node = entryPoints.get(i);
      if (node != null) {
        sb.append(node.id).append(" ");
      }
    }
    sb.append('\n');
    sb.append("nodes:\n");
    for (int i = 0; i < nodes.length(); i++) {
      Node<TItem, TDistance> node = nodes.get(i);
      if (node != null) {
        sb.append(node);
      }
    }
    sb.append('\n');
    return sb.toString();
  }
}
