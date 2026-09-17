# Azure GPU quota investigation — 2026-09-17

Outcome: **we could not obtain a GPU VM on the Azure subscription.** The
serving host for `/predict` is therefore not the Azure T4 we hoped for. This
note records what was tried, what the subscription actually allows, and the
exact blockers so it does not need repeating.

## Subscription

| Field | Value |
| --- | --- |
| Account | `domibacsro@gmail.com` |
| Subscription | `Visual Studio Enterprise Subscription` |
| Subscription id | `fb7e42a5-a721-4702-9be9-ecf545c9107d` |
| Support plan | **Free** |
| Credit | ~1550 NOK (VS Enterprise monthly Azure credit) |

Only this one subscription is visible to the account.

## What was tried, and what happened

1. **Azure CLI 2.90.0** installed on Ubuntu; `az login --use-device-code`
   succeeded.
2. **Resource providers were all unregistered** (`Microsoft.Compute`,
   `Microsoft.Network`, `Microsoft.Quota` = `NotRegistered`) — the subscription
   was fresh. Registered all four; they reached `Registered`.
3. **T4 SKUs are offered and *not* subscription-restricted** in 31 regions
   (including `northeurope`, `westeurope`, `eastus`, `westus2`). The SKU
   `Standard_NC4as_T4_v3` returns no `NotAvailableForSubscription` restriction
   in those regions.
4. **But the `Standard NCASv3_T4 Family` vCPU quota is 0 in every region.**
   Scanned 33 regions that offer the T4; every one reported `limit = 0`
   (including `northeurope`, `westeurope`, `eastus`, `swedencentral` was not
   offered at all). The `Standard NC` / `Standard NV` families carry a legacy
   default of 18 vCPUs, but those are the retired K80/M60 families, not usable.
5. **The quota API will not auto-raise it.** For `northeurope`, `westeurope`,
   `eastus` (`az quota update --resource-name "Standard NCASv3_T4 Family"
   --scope .../locations/<region> --limit-object value=4`) Azure returned:

   ```
   ERROR: (ContactSupport) Request failed.
   Code: ContactSupport
   ```

6. **The support-ticket API is gated behind a paid support plan.** Via
   `az support in-subscription tickets create` (service
   `06bfd9d3-...` "Service and subscription limits (quotas)", classification
   `e12e3d1d-...` "Compute-VM (cores-vCPUs) subscription limit increases"):

   ```
   ERROR: (InvalidSupportPlan) Your support plan type is Free. To create and
   update support tickets ... you need access to our high-tier support plans.
   ```

## Conclusion

- Region choice does not help: the GPU quota is a subscription-level allowance
  that is zero everywhere, and the increase path requires a paid support plan
  the VS Enterprise benefit subscription does not include.
- The 1550 NOK credit is still usable for **CPU** VMs and other services, just
  not for a GPU.
- A manual portal quota request (`Subscriptions → Usage + quotas → Request
  increase`) routes to the same paid-support gate; not worth pursuing for the
  competition window.

## Consequences / open options for the serving host

1. **Third-party GPU VM** (RunPod / Vast / Lambda, etc.) — instant, roughly
   $0.2–0.5/hr, and rules-compliant: the competition forbids cloud *inference
   APIs*, not running our own code on a rented VM.
2. **Azure CPU VM** — guaranteed and covered by the credit; run
   `faster-whisper` int8 on CPU. Needs benchmarking against the 60 s budget.
3. Keep IDUN for development/accuracy iteration only — it is not publicly
   reachable, so it cannot serve `/predict`.

The final call is still open; development proceeds on IDUN regardless.
