package xyz.anonymous.distbfjoin.utils;

import java.io.Serializable;
import java.util.concurrent.TimeUnit;

public class Timer implements Serializable {

  private transient long start = System.nanoTime();
  private transient long sum = 0;
  private transient boolean paused = false;

  public long duration() {
    return sum + lastLap();
  }

  public void pause() {
    sum += lastLap();
    paused = true;
  }

  public void resume() {
    paused = false;
    start = System.nanoTime();
  }

  public long lastLap() {
    if (paused) {
      return 0;
    }
    return TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start);
  }

  @Override
  public String toString() {
    return duration() + " ms";
  }
}
