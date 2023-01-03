## Next Step

- entrainement avec moins de vec_channels
- entrainement du big



## Analyses temporelles :
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