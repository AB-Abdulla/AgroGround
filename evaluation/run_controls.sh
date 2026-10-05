#!/bin/bash
# Control experiments (blank image, shuffled tiles, hard negatives) for the zero-shot 2B model,
# the mixed model, the mixed + healthy (12%) model and the RL model. Produces the ten ctl_*.json
# files scored by score.py (controls section).
# usage, from the repository root: bash evaluation/run_controls.sh <GPU>
W=./weights
G=${1:-0}
B2=./work/output_agroground_2b_recog_v2          # mixed
C2=./work/output_agroground_2b_recogneg2_v2      # mixed + healthy (12%)
RL=$W/agroground-2b-grpo-s170                    # main RL model
E=evaluation
run_ad () { python3 -u $E/eval_control_v4.py --adapter "$1" --mode "$2" --gpu "$G" --out "$3" 2>&1 | grep -v Warning | tail -1; }
run_rl () { python3 -u $E/eval_control_v4.py --adapter none --model "$RL" --mode "$1" --gpu "$G" --out "$2" 2>&1 | grep -v Warning | tail -1; }
run_ad none  blank   ctl_blank_ZERO.json
run_ad "$B2" blank   ctl_blank_B2.json
run_ad "$C2" blank   ctl_blank_C2.json
run_rl       blank   ctl_blank_RL.json
run_ad "$B2" shuffle ctl_shuf_B2.json
run_ad "$C2" shuffle ctl_shuf_C2.json
run_rl       shuffle ctl_shuf_RL.json
run_ad "$B2" hardneg ctl_hardneg_B2.json
run_ad "$C2" hardneg ctl_hardneg_C2.json
run_rl       hardneg ctl_hardneg_RL.json
echo CTL_DONE
