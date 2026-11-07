# Scenarios

Two-party negotiation scenarios in the PieArena format. Each file runs with the PieArena code as is and is scored from its own `scoring` block.

| File | Setting | Type |
|---|---|---|
| `Subscription_Box.yaml` | A beauty subscription box and a hand-cream brand negotiate a featured-product deal | Multi-issue, points |
| `Festival_Ticketing.yaml` | A music festival and a ticketing company negotiate a ticketing contract | Multi-issue, points |
| `GPU_Window.yaml` | An AI startup buys a block of GPU time from a university computing center | Single-issue price |
| `Equipment_Sale.yaml` | A chipmaker buys a used lithography scanner from another chipmaker | Single-issue price |
| `Naming_Rights.yaml` | A transit agency sells the naming rights of a light-rail station to a credit union | Multi-issue, expected value |
| `Game_Publishing.yaml` | A game publisher and an independent studio negotiate a publishing deal | Multi-issue, no ZOPA |

To run a scenario, from the repository root:

```bash
python -m piearena --scenario scenarios/Subscription_Box.yaml --rounds 6 --n-runs 6 \
  --models <model-a> <model-b> --exp-name demo
```
