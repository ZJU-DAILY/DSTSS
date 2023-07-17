package xyz.anonymous.distbfjoin.utils;

import xyz.anonymous.distbfjoin.Trajectory;
import xyz.anonymous.distbfjoin.bfjoin.MBR;
import it.unimi.dsi.fastutil.ints.IntOpenHashSet;
import xyz.anonymous.distbfjoin.Trajectory;
import xyz.anonymous.distbfjoin.bfjoin.MBR;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

public class Batch implements Iterable<Trajectory> {

  private final List<Trajectory> trajectories = new ArrayList<>();
  private MBR mbr;
  private final Set<Integer> terms = new IntOpenHashSet(); // Text summary

  public MBR getMbr() {
    return mbr;
  }

  public void add(Trajectory trajectory, MBR newMbr) {
    trajectories.add(trajectory);
    this.mbr = newMbr;
    terms.addAll(trajectory.summarize().getTerms());
  }

  public int size() {
    return trajectories.size();
  }

  public List<Trajectory> getTrajectories() {
    return trajectories;
  }

  public boolean hasTerm(int term) {
    return terms.contains(term);
  }

  @Override
  public Iterator<Trajectory> iterator() {
    return trajectories.iterator();
  }
}
