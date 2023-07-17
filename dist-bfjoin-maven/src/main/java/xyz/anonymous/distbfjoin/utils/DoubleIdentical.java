package xyz.anonymous.distbfjoin.utils;

import xyz.anonymous.distbfjoin.DoubleNormalizer;
import xyz.anonymous.distbfjoin.DoubleNormalizer;

import java.io.Serializable;
import java.util.function.DoubleUnaryOperator;

public class DoubleIdentical implements DoubleNormalizer, Serializable {

  private DoubleIdentical() {
  }

  /**
   * @param operand the operand
   * @return the operand
   */
  @Override
  public double applyAsDouble(double operand) {
    return operand;
  }

  public static DoubleIdentical DOUBLE_IDENTICAL = new DoubleIdentical();
}
