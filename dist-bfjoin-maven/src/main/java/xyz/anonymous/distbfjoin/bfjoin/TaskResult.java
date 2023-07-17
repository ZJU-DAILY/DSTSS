package xyz.anonymous.distbfjoin.bfjoin;

import java.util.PriorityQueue;
import xyz.anonymous.distbfjoin.Result;
import xyz.anonymous.distbfjoin.Result;

public class TaskResult {

  private final PriorityQueue<Result> results;
  private final long time;
  private final int count;

  public TaskResult(PriorityQueue<Result> results, long time, int count) {
    this.results = results;
    this.time = time;
    this.count = count;
  }

  public PriorityQueue<Result> getResults() {
    return results;
  }

  public long getTime() {
    return time;
  }

  public int getCount() {
    return count;
  }

}
