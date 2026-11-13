# Submission: Project 6, Task Time Detective

## Authors
Johannes Schleicher (solo)

**Team name:** Rent me if you can

**Project:** Project 6: Task Time Detective, how long does each turnaround task take?

**In short:** we did not just train a model. We rebuilt how a branch works, and that is why the numbers hold up.
- **Understand the data first.** Before any modelling we ran a data analysis with an interactive dashboard (filters
  by branch, vehicle class and date; queue vs. total time; what happens while the branch is closed; sum of tasks vs.
  total time; staffing and location type). Comparing an airport branch (3 staff, open 24/7) with a centre branch
  (1 staff, limited hours) showed the two key insights. More staff means shorter turnarounds, because tasks are shared between people. When a
  branch is closed, cars wait until it opens again. Both are modelled exactly, not learned.
- **What drives each task:** one simple linear model per task, so every effect reads in minutes. A child seat adds
  7.5 min to interior cleaning, a smoker's car 7.6 min to special cleaning, and weather affects only the exterior wash.
  The one exception is damage repair: it varies by about ±16 min and no feature in the data explains that.
- **Honest results** (each day predicted by models that never saw it): per-task WAPE **0.075** (baseline 0.55,
  "strong" < 0.2), total time error **8.1 min** (baseline 99, "good" < 15), median error 2.3 min.
- **What the customer hears:** during the day (88 % of cars) "ready at 14:35", off by 4.3 min on average. Just before
  closing (12 %) "ready tomorrow at opening", the right day for 97 % of them.

## Problem and approach
**Step 1, data analysis.** We started with an interactive dashboard over the train data to see how turnaround time
behaves per branch, staffing level and opening hours (screenshots on slide 2). It showed that the total is not the sum
of the tasks, that more staff shortens it, and that closed hours add long waits. The structure below follows from that.

**Step 2, model.** We only see which tasks were done and the total time until the car was ready. The baseline assumes
*total = sum of task times*, which is the wrong picture. The validation week, which comes with true task times, shows
a structure that holds for **all 2,791 validation turnarounds without a single violation**:

1. **Electric charging runs unattended**, in parallel with all other work.
2. **The other tasks are shared between the `staff_on_duty`** of the branch. Each person works through their tasks
   one after the other. At one-person branches (BER/NUE/STR) everything is sequential; at airports, three people
   work in parallel. Work time = max(charging, busiest person). **With a damage repair, damage check → repair →
   exterior wash run as one fixed sequence** (you wash after repairing). Without this chain, repair cars at two- and
   three-person branches seemed to wait 22 min longer than everyone else; with it, all 67 validation repair cars fit
   and their wait looks like everyone else's (median 3.5 min).
3. **The car can first wait for a free person** (queue wait, typically 0 to 30 min).
4. **Work happens only while the branch is open.** Work that does not finish before closing continues at the next
   opening. Cars dropped in the key box start at opening. This explains every total above 3 hours (up to about 2,400
   min: Saturday evening to Monday morning at branches closed on Sunday).

`total = calendar(queue wait + work time)` is computed exactly from `branch_data.csv`. Only the task times and the
queue wait are learned.

**Per-task durations:** one small ridge regression per task, fitted on the true task times in
`validation_labels.csv` (the README allows using validation however we like). Each task uses only features a branch
manager would expect (table below). Predictions are clipped to the range seen in the data, so they are never negative
or implausible.

**Queue wait:** for every train turnaround we compute exactly how many *open* minutes passed until it was ready and
subtract the predicted work time. That residual (queue wait plus any error in the work estimate) is what we learn.
Two cases:
- **Key-box cars** (returned while closed; only BER/NUE/STR on Saturday evening and Sunday) are worked off one by one
  at opening: wait = **12.0 min × cars ahead in the key box** (train: correlation 0.998, residual MAE 2.4 min). Walk-in
  customers on Monday morning do *not* queue behind this backlog (checked), so the two cases are kept apart.
- **Cars returned while open:** a gradient-boosted *median* regression (scikit-learn `HistGradientBoostingRegressor`,
  absolute-error loss, fixed seed). Its inputs are:
  - time of day, staff level, weekday and `queue_length`;
  - the car's own predicted task times and work time;
  - how busy the branch is: number of returns **and their predicted work** in the last 15, 30, 60 and 120 minutes.

  It only uses returns up to the car's own return time, never later ones. Predicting the median minimises absolute
  error. Because the elapsed time is monotone in the wait, the median also makes the right "finished before closing or
  not" call on average.
- The total is never below the car's own predicted work time: a car is not ready before its tasks are done.

A separate "finished today?" classifier for cars returned shortly before closing was tried on top of the simpler
median model and improved it (validation 13.3 → 12.3 min). On top of the boosted wait model it made held-out late
train cases worse (19.1 → 21.0 min), so it is not used.

Assumptions and choices:
- **Season is not used.** Train runs Jul to Sep, validation is the first week of October, and test is the three weeks
  after. The synthetic data shows no seasonal drift: weather mix, dirt scores and the special-cleaning share are flat
  across months. Season effects in real life should show up through `exterior_dirt` and `weather`, which we use.
- **Weather is used only where it is plausible:** exterior cleaning (rain about +1.5 min on a normal wash) and nowhere
  else. Weather did not help interior work, refuelling or the rare tasks in cross-validation.
- **Holidays are not used.** Neither 15 August (Bavaria: MUC, NUE) nor 3 October (nationwide) differs from other
  Fridays in volume or wait. The test weeks (6 to 26 October) contain no holiday in any branch's state.
- **Weekdays act only through volume and opening hours.** Friday has the most returns (about 530 vs 320 on Monday),
  but at the same rush the wait is the same Monday to Saturday. Task times do not depend on the weekday (model
  residuals by weekday below 0.5 min). Saturday's early closing and Sunday closure are handled by the calendar and the
  key-box rule.
- Rare waits above 4 hours are excluded when learning the typical walk-in wait.

## Models and why
| Model | Open-weight or closed | Where it runs | Main metric(s), leave-one-day-out on validation | Cost and time per item | Chosen? |
| --- | --- | --- | --- | --- | --- |
| Baseline (OLS of total on task flags) | open | local, numpy | total MAE 99, WAPE 0.55 | < 1 s total | no |
| Structure + ridge per task + median queue wait by (staff, hour, weekday) | open | local, numpy | total MAE 14.6, WAPE 0.077, per-task MAE mean 3.35 | about 1 s total | no, rare tasks too shrunk |
| Same, queue wait by (staff, weekday, rush), rare tasks tuned | open | local, numpy | total MAE 14.7, median AE 3.4, WAPE 0.075, per-task MAE mean 2.67 | about 1 s total | no |
| Same + key-box queue rule (12 min per car ahead) | open | local, numpy | total MAE 13.3, median AE 3.2, WAPE 0.075, per-task MAE mean 2.67 | about 1 s total | no |
| Same + "finished today?" logistic classifier near closing | open | local, numpy | total MAE 12.3, median AE 3.2, WAPE 0.075, per-task MAE mean 2.67 | about 2 s total | no |
| Key-box rule + boosted median regression for the wait (no classifier), total ≥ work | open | local, numpy + scikit-learn | total MAE 7.9, median AE 2.4, 90 % within 10 min, WAPE 0.075, per-task MAE mean 2.67 | about 2 s total | no |
| Same + repair chain (damage check → repair → exterior wash) | open | local, numpy + scikit-learn | **total MAE 8.1, median AE 2.3, 91 % within 10 min, WAPE 0.075, per-task MAE mean 2.67** | about 2 s total | **yes** |

Later train weeks the wait models never saw (wait fitted on weeks before the cut, scored after; task models as in the
submission):

| Cut | Median groups + classifier | Boosted median regression | + repair chain (chosen) |
| --- | --- | --- | --- |
| 25 Aug | 12.3 (Saturday 13.1) | 7.67 (Saturday 6.8) | **7.53** |
| 8 Sep | 12.0 (Saturday 11.0) | 6.19 (Saturday 3.8) | **5.97** |
| 15 Sep | 11.7 (Saturday 13.8) | 6.37 (Saturday 3.9) | **6.16** |

The repair chain is better on all three later-week splits but slightly worse on the single validation week
(7.91 → 8.08). That week's total error is dominated by about ten coin-flip calls at closing time, and the later weeks
hold far more data. The chain is also what the data's structure shows, so we keep it.

Learning the two rare tasks from train.csv as well was tried and dropped. We backed out about 1,100 implied repair
times from the train totals using the chain. They confirm the validation effects (damage reported about +25 min;
similar at every staff level). But adding them did not improve held-out repair error (12.57 → 12.57 to 12.65 min).
For tyre checks it made things worse (4.23 → 4.5). No other feature explains the repair time (all correlations with
the residual below 0.11), and within the same damage/class group it varies by about ±16 min. The 12.6 min error is
therefore close to the noise floor of this task.

Earlier versions of `cv.py` predicted each day on its own, which hid the Saturday key-box cars from Sunday's queue
count. It now predicts the whole week and scores only the held-out day; the numbers above use the corrected version.

Also tried and dropped:
- a fluid "backlog" queue simulation (worse than the rush count);
- FIFO and per-worker queue simulations (they over-predict: the data is not a closed queue);
- minutes-to-closing bands for the wait (no gain).

The quantile of the wait was checked on later train weeks: median (τ = 0.5 to 0.55) is best, so the boosted model
uses the absolute-error (median) loss.

## Results
**Honest check:** `starter/cv.py` predicts each validation day with task models that never saw that day.

| Metric | Baseline | Ours (leave-one-day-out) | README "strong" |
| --- | --- | --- | --- |
| Total MAE (min) | 99 | **8.1** | < 5 |
| Total median abs. error (min) | | **2.3** | |
| Share of totals within 10 min | | **91 %** | |
| Per-task WAPE | 0.55 | **0.075** | < 0.2 |
| Per-task MAE mean (macro) | 13.6 | **2.67** | |
| Negative or missing estimates | 0 | 0 | 0 |

Reproducibility:
- Task models are closed-form least squares.
- The wait model is gradient boosting with a fixed `random_state`.
- `python run.py --repeat` prints `run twice: IDENTICAL` and runs in about 2 s.
- Package versions are pinned in `requirements.txt`. The results depend on the scikit-learn version: we measured
  7.91 min with 1.9.1 for the model without the repair chain, while a 1.8.0 run of that same model reported 7.62.
  Install exactly the pinned versions.

**Per-task estimates** (validation week, where performed):

| Task | n | Our mean (min) | True mean | Our range | Main drivers (minutes) |
| --- | --- | --- | --- | --- | --- |
| quick_exterior_cleaning | 1,575 | 13.3 | 13.3 | 11.2 to 15.6 | airport branches +0.5 to +1.4, van +0.9 |
| normal_exterior_cleaning | 1,216 | 19.2 | 19.2 | 14.1 to 25.3 | van +2.3, exterior dirt +1.8 per level, rain +1.5, snow +1.2 |
| special_cleaning | 277 | 30.0 | 30.0 | 27.5 to 38.2 | smoking +7.6, premium +2.6, pet +2.0, child seat +1.5 |
| interior_cleaning | 2,791 | 16.4 | 16.4 | 10.4 to 31.4 | child seat +7.5, interior dirt +2.2 per level, van +1.8 |
| damage_check | 1,394 | 2.4 | 2.4 | 2.3 to 2.8 | damage reported +0.3 |
| refuel | 410 | 13.5 | 13.5 | 5.3 to 18.9 | branch: FRA +5.7, MUC +4.7, NUE -4.1 (vs BER); van +3.9 |
| electric_charging | 526 | 24.8 | 24.8 | 8.7 to 40.1 | battery to fill: +21 min linear and +12 min quadratic (0 to 100 %) |
| quick_damage_repair | 71 | 68.6 | 68.6 | 54.1 to 79.5 | damage reported +17.9, SUV +5.4 |
| tyre_check | 29 | 26.1 | 26.1 | 17.9 to 45.3 | staff 2: -13, staff 3: -27 (vs one-person branch about 45 min) |

**Queue wait in the data** (train, median minutes) grows with how many cars came in recently. This is what the boosted
model picks up, together with the predicted work of those cars:
- one-person branch: 1 → 12 → 23 → 33 → 44 min for 1 to 5 cars in the last 30 min;
- two-person branch: 1 → 1 → 7 → 12 → 17 min;
- three-person branch: 1 → 1 → 1 → 4 → 9 min;
- key-box cars: 12.0 min per car ahead, so the 12th Sunday drop-off at a one-person branch waits about 2 h 10 min on
  Monday before work starts.

**What it means for the customer ("your car is ready at …"):** return time + predicted total = ready time. Its accuracy
depends on when the car comes back (leave-one-day-out on validation):

| Cars | n | Total MAE (min) | Share of all error |
| --- | --- | --- | --- |
| all | 2,791 | 8.1 | 100 % |
| returned in the last 2 h before closing | 339 (12 %) | 35.5 | 53 % |
| all others (returned during the day) | 2,452 (88 %) | **4.3** | 47 % |
| key-box cars (returned while closed) | 47 | 2.6 | |

Half of the error comes from the 12 % of cars returned shortly before closing. Their exact ready minute rarely matters
because they are not rented out again that evening. What matters is *which day* they are ready, and that is right for
97 % of them. A customer-facing message would therefore be "ready at 14:35" during the day and "ready tomorrow at
opening" near closing. All cars are still predicted and scored. The 4.3 min is not a claim to the README's "strong"
level, which applies to all cars.

**Limits, honestly:**
- Errors by day of the validation week (leave-one-day-out): Mon to Fri 3.5 to 9 min, Sun 3.6 min, but
  **Saturday 21 min**.
  Two Saturday cars at centre branches cause most of that:
  - STR 14:59 was predicted ready at about 45 min but was ready on Monday (2,434 min);
  - BER 15:19 was predicted to wait until Monday but was ready in 40 min.

  A car that misses the 16:00 closing waits until Monday (about 2,400 min instead of about 750 on a weekday), so a
  single wrong call there is very expensive. On the later train weeks Saturday is fine (3.8 to 6.8 min).
- The wait is partly random, so calls right at closing time cannot all be right. That is why the total reaches "good"
  (< 15) but not "strong" (< 5).
- The learned wait is a *correction*, not a pure physical queue: it also absorbs errors in the work estimate (in about
  9 % of cars it is slightly negative, at most -3.7 min). The total is floored at the predicted work time, so no car is
  ever ready before its own tasks are done.
- The two rare tasks have few labelled examples (71 and 29). Their effects are cross-checked against train.csv:
  one-person branches imply about 46 min for tyre checks and about 66 min for repairs, consistent with validation.
- The task models are learned from one week. If staffing or opening hours change, `branch_data.csv` carries that
  through the structure, but the per-task times would need re-checking.

## How to run
```
cd Project_6
python3 -m venv .venv && . .venv/bin/activate    # Python 3.11+ (tested with 3.14)
pip install -r requirements.txt                  # numpy, pandas, scikit-learn (pinned versions)
cd starter
python run.py --repeat                           # fit + predict validation, checks two runs are identical
python evaluate.py predictions.csv ../data/validation_labels.csv --task-truth ../data/validation_labels.csv \
    --flags ../data/validation.csv --train-truth ../data/validation_labels.csv
python cv.py                                     # honest leave-one-day-out numbers from the table above
```
`solution.py` reads `data/validation.csv` and `data/validation_labels.csv` itself to learn the task times. No network,
no secrets, no environment variables.

## Demo
Slides (5 pages): [`submission/slides.pdf`](submission/slides.pdf)
1. Rent me if you can: who I am.
2. First step: understand the data (airport vs. centre branch, staff and opening hours).
3. What drives each task (child seat, smoker's car, weather; damage repair as the exception).
4. Honest results (baseline vs. ours vs. README targets).
5. What the customer hears ("ready at 14:35" during the day, "ready tomorrow at opening" near closing).

## Next steps
- Model the end-of-day decision directly ("will it finish today?") as a probability, and report both a ready time and
  a risk to the customer.
- Use the structure for staff planning: simulate how an extra person at a one-person branch between 17:00 and 19:00
  changes overnight spill-over.
- With a full year of data, check whether dirt scores fully capture seasonal effects (winter, road salt) or a season
  term is needed.
