package xyz.anonymous.distbfjoin.utils;


import xyz.anonymous.distbfjoin.DoubleNormalizer;
import xyz.anonymous.distbfjoin.DoubleNormalizer;

import java.io.Serializable;
import java.util.function.DoubleUnaryOperator;

public class DoubleSigmoid implements DoubleNormalizer, Serializable {

  double gamma;

  public DoubleSigmoid(double gamma) {
    this.gamma = gamma;
  }

  /**
   * @param operand the operand
   * @return value of Sigmoid function
   */
  @Override
  public double applyAsDouble(double operand) {
    return 2 - 2.0 / (1 + Math.exp(-gamma * operand));
  }
}
