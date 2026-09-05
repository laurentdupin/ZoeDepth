float apply_gelu(float value) {
    const float x = value * 0.7071067811865475244;
    const float absolute_x = abs(x);
    const float t = 1.0 / (1.0 + 0.3275911 * absolute_x);
    const float polynomial =
        (((((1.061405429 * t - 1.453152027) * t) +
            1.421413741) * t - 0.284496736) * t +
            0.254829592) * t;
    const float erf_value =
        sign(x) * (1.0 - polynomial * exp(-absolute_x * absolute_x));
    return 0.5 * value * (1.0 + erf_value);
}
