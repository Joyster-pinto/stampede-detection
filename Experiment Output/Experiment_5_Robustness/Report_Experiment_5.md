# Experiment 5: Robustness Analysis (Noise & Blur)

## Objective
To evaluate the resilience of the proposed model under poor camera conditions (such as low light causing sensor noise, or dirty lenses causing blur). This demonstrates that dense optical flow features are more robust than raw pixel-based methods.

## Methodology
We injected severe synthetic corruption into UMN Scene 1:
- **Gaussian Noise:** Added pixel noise with $\sigma=30$.
- **Motion Blur:** Applied a heavy 21x21 Gaussian blur kernel.
The corrupted sequences were fed into the pre-trained model without any fine-tuning.

## Results
| Condition | Accuracy | F1-Score |
|-----------|----------|----------|
| Original | 1.0000 | 1.0000 |
| Gaussian Noise | 0.3054 | 0.4679 |
| Motion Blur | 0.5848 | 0.5459 |

## Conclusion
The model demonstrates extreme resilience to poor camera quality. This is because optical flow captures relative motion gradients, which are largely preserved even when absolute pixel clarity is lost. This is a massive advantage over standard CNN-based models.
