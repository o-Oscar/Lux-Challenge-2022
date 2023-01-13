## Analyses temporelles :

#### evaluation sur 100 games

factory_survivor_light_32_vec : 3min25, 1.108083333333333 +/- 0.07349503219609033
imitator_light_32_vec : 3min17, 1.274
imitator_light_8_vec : 3min34, 1.24
imitator_super_light_8_vec : 3min33, 1.307
imitator_super_light_deep_8_vec : 3min39, 1.306
baseline : 40min, 2

```
(luxai) PS C:\Users\stani\Documents\MVA\S1\RL\Lux-Challenge-2022> python -m learning.evaluate --bot_type factory_survivor_light --vec_chan 32 --name 32_vec --max_length 150 --num_games 100
factory_survivor detected, set reward_generator to FactorySurvivorRewardGenerator

100%|████████████████████████████████████████████████████████████████████████████████| 100/100 [03:20<00:00,  2.01s/it]
Mean transfer per factory and per game : 1.1230833333333332 +/- 0.07409807467391268
Mean time per per game : 1.998320837020874 +/- 0.10722089748353583
(luxai) PS C:\Users\stani\Documents\MVA\S1\RL\Lux-Challenge-2022> python -m learning.evaluate --bot_type imitator_light --vec_chan 32 --name 32_vec --max_length 150 --num_games 100

imitator detected, set reward_generator to ImitationRewardGenerator

100%|████████████████████████████████████████████████████████████████████████████████| 100/100 [03:24<00:00,  2.04s/it]
Mean transfer per factory and per game : 1.3271666666666666 +/- 0.07923063294081316
Mean time per per game : 2.029709243774414 +/- 0.09398724920062117
(luxai) PS C:\Users\stani\Documents\MVA\S1\RL\Lux-Challenge-2022> python -m learning.evaluate --bot_type imitator_light --vec_chan 8 --name 8_vec --max_length 150 --num_games 100

imitator detected, set reward_generator to ImitationRewardGenerator

100%|████████████████████████████████████████████████████████████████████████████████| 100/100 [03:20<00:00,  2.00s/it]
Mean transfer per factory and per game : 1.3246666666666667 +/- 0.07856496484340617
Mean time per per game : 1.9905843186378478 +/- 0.07620723016969638
(luxai) PS C:\Users\stani\Documents\MVA\S1\RL\Lux-Challenge-2022> python -m learning.evaluate --bot_type imitator_super_light --vec_chan 8 --name 8_vec --max_length 150 --num_games 100

imitator detected, set reward_generator to ImitationRewardGenerator

super_light detected, change grid_kernel_size to 11

100%|████████████████████████████████████████████████████████████████████████████████| 100/100 [03:31<00:00,  2.11s/it]
Mean transfer per factory and per game : 1.3642500000000002 +/- 0.08005650431091711
Mean time per per game : 2.1030028295516967 +/- 0.09182390066842566
(luxai) PS C:\Users\stani\Documents\MVA\S1\RL\Lux-Challenge-2022> python -m learning.evaluate --bot_type imitator_super_light_deep --vec_chan 8 --name 8_vec --max_length 150 --num_games 100

imitator detected, set reward_generator to ImitationRewardGenerator

super_light detected, change grid_kernel_size to 11

light_deep detected, change inside_layers_nb to 1

100%|████████████████████████████████████████████████████████████████████████████████| 100/100 [03:30<00:00,  2.11s/it]
Mean transfer per factory and per game : 1.2948333333333335 +/- 0.08079632316591565
Mean time per per game : 2.096092176437378 +/- 0.08731581672289286

(luxai) PS C:\Users\stani\Documents\MVA\S1\RL\Lux-Challenge-2022> python -m learning.evaluate_baseline --max_length 150 --num_games 100
100%|████████████████████████████████████████████████████████████████████████| 100/100 [27:20<00:00, 16.40s/it]
Mean transfer per factory and per game : 1.8527500000000001 +/- 0.0663819114879257
Mean time per per game : 16.388474225997925 +/- 0.1458688346587431
(luxai) PS C:\Users\stani\Documents\MVA\S1\RL\Lux-Challenge-2022> python -m learning.evaluate_baseline --max_length 150 --num_games 100 --null_agent
100%|████████████████████████████████████████████████████████████████████████| 100/100 [02:40<00:00,  1.60s/it]
Mean transfer per factory and per game : 0.0 +/- 0.0
Mean time per per game : 1.5906530833244323 +/- 0.04644516451991314
```


### model
temps pour 5*1200 (default is v32, l_150)

light : 80 (but in fact 93)
light_t50 : 95
light_t300 : 80
light_t1000 : 100

light_v8 : 75
light_v1 : 75

super_light : 68
super_light_v8 : 62

big_v8 : 80



### batch_size/learning_batch_size

light ?
5*1200 => 72s => 140s/12000
10*600 => 95s => 190s/12000
5*600 => 37s => 148s/12000
2*1200 => 25s => 125s/12000
2*900 => 18s => 120s/12000
3*1000 => 40sec => 160s/12000