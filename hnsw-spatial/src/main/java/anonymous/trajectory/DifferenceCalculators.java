package anonymous.trajectory;

import java.util.function.BinaryOperator;

public final class DifferenceCalculators {

  static class IntDifferenceCalculator implements BinaryOperator<Integer> {

    @Override
    public Integer apply(Integer a, Integer b) {
      return Math.abs(a - b);
    }
  }

  static class LongDifferenceCalculator implements BinaryOperator<Long> {

    @Override
    public Long apply(Long a, Long b) {
      return Math.abs(a - b);
    }
  }

  static class FloatDifferenceCalculator implements BinaryOperator<Float> {

    @Override
    public Float apply(Float a, Float b) {
      return Math.abs(a - b);
    }
  }

  static class DoubleDifferenceCalculator implements BinaryOperator<Double> {

    @Override
    public Double apply(Double a, Double b) {
      return Math.abs(a - b);
    }
  }

  public static final BinaryOperator<Integer> INT_DIFFERENCE_CALCULATOR = new IntDifferenceCalculator();
  public static final BinaryOperator<Long> LONG_DIFFERENCE_CALCULATOR = new LongDifferenceCalculator();
  public static final BinaryOperator<Float> FLOAT_DIFFERENCE_CALCULATOR = new FloatDifferenceCalculator();
  public static final BinaryOperator<Double> DOUBLE_DIFFERENCE_CALCULATOR = new DoubleDifferenceCalculator();
}
