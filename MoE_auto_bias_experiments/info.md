# Формулы bias:

| Версия | Формула $\Delta b_i$ |
| :----- | :---------- |
| **v4** | $\Delta b_i = \frac{\sigma(\bar{z}_l)}{N/K} \cdot \hat{s}_i$ |
| **v5** | $\Delta b_i = \max(\bar{z}_l^1 - \bar{z}_l^K,\ 10^{-4}) \cdot \hat{s}_i$ |
| **v6** | $\Delta b_i = \frac{4\,\sigma(z_l)}{N} \cdot \hat{s}_i$ |
| **v7** | $\Delta b_i = \frac{Q_{75}(z_l) - Q_{25}(z_l)}{N} \cdot \hat{s}_i$ |
| **v8** | $\Delta b_i = \frac{\bar{z}_l^1 - \bar{z}_l^K}{N/K} \cdot \hat{s}_i$ |
| **v9** | $\Delta b_i = \frac{\sigma(\bar{z}_l)}{N/K} \cdot \mathrm{sign}(\hat{s}_i) \cdot \hat{s}_i^{2}$ |
| **v10** | $\Delta b_i = \frac{\sigma(\bar{z}_l)}{N/K} \cdot \tilde{s}_i$, где $\tilde{s}_i = \hat{s}_i$ для топ-$K$ и анти-топ-$K$ экспертов, иначе $0$ |


# Эксп с критерием:

| Режим | Формула $S_i$ |
| :---- | :---- |
| **standard** | $S_i = -\sum_t \left(\frac{\partial L}{\partial o_t}\right)^\top w_{t,i} \cdot e_i(h_t)$ |
| **corrected** | $S_i = -\sum_t \left(\frac{\partial L}{\partial o_t}\right)^\top w_{t,i} \cdot (e_i(h_t) - o_t)$ (по сути $\partial L / \partial b_i$) |

$o_t$ — выход MoE-блока.


# Агрегация по таблицам:

## Qwen-1.5

| Таблица | 🟢 % | 🔴 % | ⬜ % |
| :------ | ---: | ---: | ---: |
| Math · Standard      | 71% | 20% |  7% |
| Math · Corrected     | 67% | 22% |  9% |
| Physics · Standard   | 32% | 52% | 15% |
| Physics · Corrected  | 30% | 63% |  5% |
| History · Standard   | 40% | 42% | 16% |
| History · Corrected  | 49% | 35% | 15% |
| Biology · Standard   | 26% | 63% |  9% |
| Biology · Corrected  | 15% | 58% | 25% |
| Chemistry · Standard | 39% | 46% | 14% |
| Chemistry · Corrected| 46% | 39% | 14% |

## Qwen-3

| Таблица | 🟢 % | 🔴 % | ⬜ % |
| :------ | ---: | ---: | ---: |
| Math · Standard      | 37% | 34% | 28% |
| Math · Corrected     | 38% | 38% | 23% |
| Physics · Standard   | 13% | 57% | 29% |
| Physics · Corrected  | 15% | 53% | 30% |
| History · Standard   | 20% | 49% | 29% |
| History · Corrected  | 13% | 47% | 39% |
| Biology · Standard   | 44% | 25% | 30% |
| Biology · Corrected  | 31% | 41% | 26% |
| Chemistry · Standard |  3% | 77% | 19% |
| Chemistry · Corrected|  1% | 76% | 22% |

### По bias (сквозные по всем датасетам)

**Qwen-1.5**

| Bias | 🟢 % | 🔴 % | ⬜ % |
| :--- | ---: | ---: | ---: |
| v4  | 48% | 35% | 15% |
| v5  | 36% | 57% |  5% |
| v6  | 49% | 39% | 11% |
| v7  | 50% | 35% | 15% |
| v8  | 45% | 41% | 12% |
| v9  | 43% | 41% | 15% |
| v10 | 39% | 45% | 15% |

**Qwen-3**

| Bias | 🟢 % | 🔴 % | ⬜ % |
| :--- | ---: | ---: | ---: |
| v4  | 28% | 48% | 23% |
| v5  |  7% | 89% |  3% |
| v6  | 28% | 29% | 42% |
| v7  | 16% | 40% | 43% |
| v8  | 22% | 46% | 30% |
| v9  | 25% | 45% | 28% |
| v10 | 29% | 42% | 28% |

# Все цифры:

## Qwen-1.5

### Math · Standard

| Bias | Loss | col_math | abstr_alg | elem_math | hs_math | all_part |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BEFORE** | — | 35.00 | 38.00 | 40.21 | 30.37 | 55.67 |
| **v4** | top_k | 40.00<br><sub style="color: #2da44e">+5.00</sub> | 39.00<br><sub style="color: #2da44e">+1.00</sub> | 41.01<br><sub style="color: #2da44e">+0.80</sub> | 30.74<br><sub style="color: #2da44e">+0.37</sub> | 56.33<br><sub style="color: #2da44e">+0.66</sub> |
| **v4** | top_k_x2 | 39.00<br><sub style="color: #2da44e">+4.00</sub> | 40.00<br><sub style="color: #2da44e">+2.00</sub> | 41.01<br><sub style="color: #2da44e">+0.80</sub> | 31.11<br><sub style="color: #2da44e">+0.74</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v4** | all | 38.00<br><sub style="color: #2da44e">+3.00</sub> | 39.00<br><sub style="color: #2da44e">+1.00</sub> | 40.74<br><sub style="color: #2da44e">+0.53</sub> | 31.48<br><sub style="color: #2da44e">+1.11</sub> | 55.67 |
| **v5** | top_k | 39.00<br><sub style="color: #2da44e">+4.00</sub> | 31.00<br><sub style="color: #cf222e">-7.00</sub> | 36.24<br><sub style="color: #cf222e">-3.97</sub> | 34.07<br><sub style="color: #2da44e">+3.70</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v5** | top_k_x2 | 37.00<br><sub style="color: #2da44e">+2.00</sub> | 37.00<br><sub style="color: #cf222e">-1.00</sub> | 35.98<br><sub style="color: #cf222e">-4.23</sub> | 30.74<br><sub style="color: #2da44e">+0.37</sub> | 54.33<br><sub style="color: #cf222e">-1.34</sub> |
| **v5** | all | 40.00<br><sub style="color: #2da44e">+5.00</sub> | 32.00<br><sub style="color: #cf222e">-6.00</sub> | 38.36<br><sub style="color: #cf222e">-1.85</sub> | 32.96<br><sub style="color: #2da44e">+2.59</sub> | 52.67<br><sub style="color: #cf222e">-3.00</sub> |
| **v6** | top_k | 40.00<br><sub style="color: #2da44e">+5.00</sub> | 39.00<br><sub style="color: #2da44e">+1.00</sub> | 40.74<br><sub style="color: #2da44e">+0.53</sub> | 30.74<br><sub style="color: #2da44e">+0.37</sub> | 56.00<br><sub style="color: #2da44e">+0.33</sub> |
| **v6** | top_k_x2 | 38.00<br><sub style="color: #2da44e">+3.00</sub> | 40.00<br><sub style="color: #2da44e">+2.00</sub> | 41.53<br><sub style="color: #2da44e">+1.32</sub> | 32.22<br><sub style="color: #2da44e">+1.85</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v6** | all | 39.00<br><sub style="color: #2da44e">+4.00</sub> | 39.00<br><sub style="color: #2da44e">+1.00</sub> | 40.74<br><sub style="color: #2da44e">+0.53</sub> | 32.59<br><sub style="color: #2da44e">+2.22</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v7** | top_k | 38.00<br><sub style="color: #2da44e">+3.00</sub> | 41.00<br><sub style="color: #2da44e">+3.00</sub> | 40.74<br><sub style="color: #2da44e">+0.53</sub> | 30.74<br><sub style="color: #2da44e">+0.37</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v7** | top_k_x2 | 37.00<br><sub style="color: #2da44e">+2.00</sub> | 40.00<br><sub style="color: #2da44e">+2.00</sub> | 39.68<br><sub style="color: #cf222e">-0.53</sub> | 30.74<br><sub style="color: #2da44e">+0.37</sub> | 55.67 |
| **v7** | all | 38.00<br><sub style="color: #2da44e">+3.00</sub> | 40.00<br><sub style="color: #2da44e">+2.00</sub> | 41.27<br><sub style="color: #2da44e">+1.06</sub> | 30.37 | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v8** | top_k | 40.00<br><sub style="color: #2da44e">+5.00</sub> | 40.00<br><sub style="color: #2da44e">+2.00</sub> | 40.48<br><sub style="color: #2da44e">+0.27</sub> | 32.96<br><sub style="color: #2da44e">+2.59</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v8** | top_k_x2 | 38.00<br><sub style="color: #2da44e">+3.00</sub> | 39.00<br><sub style="color: #2da44e">+1.00</sub> | 41.01<br><sub style="color: #2da44e">+0.80</sub> | 30.37 | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v8** | all | 39.00<br><sub style="color: #2da44e">+4.00</sub> | 40.00<br><sub style="color: #2da44e">+2.00</sub> | 41.01<br><sub style="color: #2da44e">+0.80</sub> | 32.22<br><sub style="color: #2da44e">+1.85</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v9** | top_k | 38.00<br><sub style="color: #2da44e">+3.00</sub> | 39.00<br><sub style="color: #2da44e">+1.00</sub> | 40.48<br><sub style="color: #2da44e">+0.27</sub> | 30.37 | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v9** | top_k_x2 | 39.00<br><sub style="color: #2da44e">+4.00</sub> | 40.00<br><sub style="color: #2da44e">+2.00</sub> | 40.74<br><sub style="color: #2da44e">+0.53</sub> | 31.48<br><sub style="color: #2da44e">+1.11</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v9** | all | 38.00<br><sub style="color: #2da44e">+3.00</sub> | 40.00<br><sub style="color: #2da44e">+2.00</sub> | 40.48<br><sub style="color: #2da44e">+0.27</sub> | 31.11<br><sub style="color: #2da44e">+0.74</sub> | 55.67 |
| **v10** | top_k | 39.00<br><sub style="color: #2da44e">+4.00</sub> | 38.00 | 39.68<br><sub style="color: #cf222e">-0.53</sub> | 31.11<br><sub style="color: #2da44e">+0.74</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v10** | top_k_x2 | 39.00<br><sub style="color: #2da44e">+4.00</sub> | 40.00<br><sub style="color: #2da44e">+2.00</sub> | 40.48<br><sub style="color: #2da44e">+0.27</sub> | 31.85<br><sub style="color: #2da44e">+1.48</sub> | 56.00<br><sub style="color: #2da44e">+0.33</sub> |
| **v10** | all | 39.00<br><sub style="color: #2da44e">+4.00</sub> | 39.00<br><sub style="color: #2da44e">+1.00</sub> | 40.48<br><sub style="color: #2da44e">+0.27</sub> | 30.37 | 56.00<br><sub style="color: #2da44e">+0.33</sub> |

### Physics · Standard

| Bias | Loss | col_phys | conc_phys | hs_phys | all_part |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BEFORE** | — | 32.35 | 51.91 | 37.09 | 55.67 |
| **v4** | top_k | 35.29<br><sub style="color: #2da44e">+2.94</sub> | 51.06<br><sub style="color: #cf222e">-0.85</sub> | 37.09 | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v4** | top_k_x2 | 35.29<br><sub style="color: #2da44e">+2.94</sub> | 51.49<br><sub style="color: #cf222e">-0.42</sub> | 35.76<br><sub style="color: #cf222e">-1.33</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v4** | all | 35.29<br><sub style="color: #2da44e">+2.94</sub> | 50.64<br><sub style="color: #cf222e">-1.27</sub> | 37.09 | 55.67 |
| **v5** | top_k | 36.27<br><sub style="color: #2da44e">+3.92</sub> | 48.51<br><sub style="color: #cf222e">-3.40</sub> | 37.09 | 52.00<br><sub style="color: #cf222e">-3.67</sub> |
| **v5** | top_k_x2 | 36.27<br><sub style="color: #2da44e">+3.92</sub> | 47.66<br><sub style="color: #cf222e">-4.25</sub> | 35.76<br><sub style="color: #cf222e">-1.33</sub> | 51.33<br><sub style="color: #cf222e">-4.34</sub> |
| **v5** | all | 33.33<br><sub style="color: #2da44e">+0.98</sub> | 46.38<br><sub style="color: #cf222e">-5.53</sub> | 33.11<br><sub style="color: #cf222e">-3.98</sub> | 51.33<br><sub style="color: #cf222e">-4.34</sub> |
| **v6** | top_k | 35.29<br><sub style="color: #2da44e">+2.94</sub> | 51.06<br><sub style="color: #cf222e">-0.85</sub> | 36.42<br><sub style="color: #cf222e">-0.67</sub> | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v6** | top_k_x2 | 34.31<br><sub style="color: #2da44e">+1.96</sub> | 51.49<br><sub style="color: #cf222e">-0.42</sub> | 37.09 | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v6** | all | 35.29<br><sub style="color: #2da44e">+2.94</sub> | 51.06<br><sub style="color: #cf222e">-0.85</sub> | 37.75<br><sub style="color: #2da44e">+0.66</sub> | 56.00<br><sub style="color: #2da44e">+0.33</sub> |
| **v7** | top_k | 34.31<br><sub style="color: #2da44e">+1.96</sub> | 50.21<br><sub style="color: #cf222e">-1.70</sub> | 37.09 | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v7** | top_k_x2 | 34.31<br><sub style="color: #2da44e">+1.96</sub> | 50.64<br><sub style="color: #cf222e">-1.27</sub> | 37.09 | 55.67 |
| **v7** | all | 32.35 | 51.06<br><sub style="color: #cf222e">-0.85</sub> | 34.44<br><sub style="color: #cf222e">-2.65</sub> | 56.33<br><sub style="color: #2da44e">+0.66</sub> |
| **v8** | top_k | 33.33<br><sub style="color: #2da44e">+0.98</sub> | 50.21<br><sub style="color: #cf222e">-1.70</sub> | 37.09 | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v8** | top_k_x2 | 33.33<br><sub style="color: #2da44e">+0.98</sub> | 51.49<br><sub style="color: #cf222e">-0.42</sub> | 36.42<br><sub style="color: #cf222e">-0.67</sub> | 55.67 |
| **v8** | all | 33.33<br><sub style="color: #2da44e">+0.98</sub> | 50.21<br><sub style="color: #cf222e">-1.70</sub> | 38.41<br><sub style="color: #2da44e">+1.32</sub> | 56.00<br><sub style="color: #2da44e">+0.33</sub> |
| **v9** | top_k | 34.31<br><sub style="color: #2da44e">+1.96</sub> | 50.64<br><sub style="color: #cf222e">-1.27</sub> | 37.75<br><sub style="color: #2da44e">+0.66</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v9** | top_k_x2 | 35.29<br><sub style="color: #2da44e">+2.94</sub> | 51.49<br><sub style="color: #cf222e">-0.42</sub> | 37.09 | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v9** | all | 34.31<br><sub style="color: #2da44e">+1.96</sub> | 50.64<br><sub style="color: #cf222e">-1.27</sub> | 36.42<br><sub style="color: #cf222e">-0.67</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v10** | top_k | 35.29<br><sub style="color: #2da44e">+2.94</sub> | 51.06<br><sub style="color: #cf222e">-0.85</sub> | 37.09 | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v10** | top_k_x2 | 34.31<br><sub style="color: #2da44e">+1.96</sub> | 51.06<br><sub style="color: #cf222e">-0.85</sub> | 38.41<br><sub style="color: #2da44e">+1.32</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v10** | all | 35.29<br><sub style="color: #2da44e">+2.94</sub> | 50.21<br><sub style="color: #cf222e">-1.70</sub> | 36.42<br><sub style="color: #cf222e">-0.67</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |

### History · Standard

| Bias | Loss | hs_world | hs_euro | hs_us | prehistory | all_part |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BEFORE** | — | 70.89 | 68.48 | 69.12 | 65.43 | 55.67 |
| **v4** | top_k | 71.73<br><sub style="color: #2da44e">+0.84</sub> | 68.48 | 70.10<br><sub style="color: #2da44e">+0.98</sub> | 63.89<br><sub style="color: #cf222e">-1.54</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v4** | top_k_x2 | 71.31<br><sub style="color: #2da44e">+0.42</sub> | 67.88<br><sub style="color: #cf222e">-0.60</sub> | 71.08<br><sub style="color: #2da44e">+1.96</sub> | 64.20<br><sub style="color: #cf222e">-1.23</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v4** | all | 71.73<br><sub style="color: #2da44e">+0.84</sub> | 67.88<br><sub style="color: #cf222e">-0.60</sub> | 71.08<br><sub style="color: #2da44e">+1.96</sub> | 65.12<br><sub style="color: #cf222e">-0.31</sub> | 56.00<br><sub style="color: #2da44e">+0.33</sub> |
| **v5** | top_k | 72.15<br><sub style="color: #2da44e">+1.26</sub> | 65.45<br><sub style="color: #cf222e">-3.03</sub> | 69.12 | 64.20<br><sub style="color: #cf222e">-1.23</sub> | 57.67<br><sub style="color: #2da44e">+2.00</sub> |
| **v5** | top_k_x2 | 73.00<br><sub style="color: #2da44e">+2.11</sub> | 66.67<br><sub style="color: #cf222e">-1.81</sub> | 68.63<br><sub style="color: #cf222e">-0.49</sub> | 63.58<br><sub style="color: #cf222e">-1.85</sub> | 61.00<br><sub style="color: #2da44e">+5.33</sub> |
| **v5** | all | 72.15<br><sub style="color: #2da44e">+1.26</sub> | 64.24<br><sub style="color: #cf222e">-4.24</sub> | 71.08<br><sub style="color: #2da44e">+1.96</sub> | 66.05<br><sub style="color: #2da44e">+0.62</sub> | 60.67<br><sub style="color: #2da44e">+5.00</sub> |
| **v6** | top_k | 71.31<br><sub style="color: #2da44e">+0.42</sub> | 68.48 | 71.08<br><sub style="color: #2da44e">+1.96</sub> | 64.20<br><sub style="color: #cf222e">-1.23</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v6** | top_k_x2 | 70.89 | 68.48 | 70.10<br><sub style="color: #2da44e">+0.98</sub> | 64.20<br><sub style="color: #cf222e">-1.23</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v6** | all | 71.31<br><sub style="color: #2da44e">+0.42</sub> | 67.88<br><sub style="color: #cf222e">-0.60</sub> | 71.08<br><sub style="color: #2da44e">+1.96</sub> | 65.12<br><sub style="color: #cf222e">-0.31</sub> | 55.67 |
| **v7** | top_k | 70.89 | 69.09<br><sub style="color: #2da44e">+0.61</sub> | 69.61<br><sub style="color: #2da44e">+0.49</sub> | 64.81<br><sub style="color: #cf222e">-0.62</sub> | 56.00<br><sub style="color: #2da44e">+0.33</sub> |
| **v7** | top_k_x2 | 71.31<br><sub style="color: #2da44e">+0.42</sub> | 69.09<br><sub style="color: #2da44e">+0.61</sub> | 69.61<br><sub style="color: #2da44e">+0.49</sub> | 64.81<br><sub style="color: #cf222e">-0.62</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v7** | all | 70.89 | 69.09<br><sub style="color: #2da44e">+0.61</sub> | 69.61<br><sub style="color: #2da44e">+0.49</sub> | 65.12<br><sub style="color: #cf222e">-0.31</sub> | 56.00<br><sub style="color: #2da44e">+0.33</sub> |
| **v8** | top_k | 71.73<br><sub style="color: #2da44e">+0.84</sub> | 68.48 | 70.59<br><sub style="color: #2da44e">+1.47</sub> | 64.51<br><sub style="color: #cf222e">-0.92</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v8** | top_k_x2 | 71.31<br><sub style="color: #2da44e">+0.42</sub> | 67.88<br><sub style="color: #cf222e">-0.60</sub> | 70.59<br><sub style="color: #2da44e">+1.47</sub> | 64.51<br><sub style="color: #cf222e">-0.92</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v8** | all | 70.89 | 68.48 | 71.08<br><sub style="color: #2da44e">+1.96</sub> | 63.89<br><sub style="color: #cf222e">-1.54</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v9** | top_k | 71.31<br><sub style="color: #2da44e">+0.42</sub> | 68.48 | 70.59<br><sub style="color: #2da44e">+1.47</sub> | 64.81<br><sub style="color: #cf222e">-0.62</sub> | 55.67 |
| **v9** | top_k_x2 | 71.31<br><sub style="color: #2da44e">+0.42</sub> | 68.48 | 69.61<br><sub style="color: #2da44e">+0.49</sub> | 64.81<br><sub style="color: #cf222e">-0.62</sub> | 55.67 |
| **v9** | all | 70.89 | 67.88<br><sub style="color: #cf222e">-0.60</sub> | 70.10<br><sub style="color: #2da44e">+0.98</sub> | 64.51<br><sub style="color: #cf222e">-0.92</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v10** | top_k | 72.15<br><sub style="color: #2da44e">+1.26</sub> | 68.48 | 70.59<br><sub style="color: #2da44e">+1.47</sub> | 64.20<br><sub style="color: #cf222e">-1.23</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v10** | top_k_x2 | 70.46<br><sub style="color: #cf222e">-0.43</sub> | 67.88<br><sub style="color: #cf222e">-0.60</sub> | 69.61<br><sub style="color: #2da44e">+0.49</sub> | 64.20<br><sub style="color: #cf222e">-1.23</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v10** | all | 70.46<br><sub style="color: #cf222e">-0.43</sub> | 67.27<br><sub style="color: #cf222e">-1.21</sub> | 70.59<br><sub style="color: #2da44e">+1.47</sub> | 64.20<br><sub style="color: #cf222e">-1.23</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |

### Biology · Standard

| Bias | Loss | col_bio | hs_bio | all_part |
| :--- | :--- | :--- | :--- | :--- |
| **BEFORE** | — | 58.33 | 68.06 | 55.67 |
| **v4** | top_k | 59.72<br><sub style="color: #2da44e">+1.39</sub> | 68.06 | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v4** | top_k_x2 | 59.03<br><sub style="color: #2da44e">+0.70</sub> | 67.74<br><sub style="color: #cf222e">-0.32</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v4** | all | 58.33 | 67.42<br><sub style="color: #cf222e">-0.64</sub> | 54.33<br><sub style="color: #cf222e">-1.34</sub> |
| **v5** | top_k | 55.56<br><sub style="color: #cf222e">-2.77</sub> | 65.48<br><sub style="color: #cf222e">-2.58</sub> | 51.67<br><sub style="color: #cf222e">-4.00</sub> |
| **v5** | top_k_x2 | 59.03<br><sub style="color: #2da44e">+0.70</sub> | 67.10<br><sub style="color: #cf222e">-0.96</sub> | 53.33<br><sub style="color: #cf222e">-2.34</sub> |
| **v5** | all | 57.64<br><sub style="color: #cf222e">-0.69</sub> | 63.55<br><sub style="color: #cf222e">-4.51</sub> | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v6** | top_k | 59.03<br><sub style="color: #2da44e">+0.70</sub> | 67.74<br><sub style="color: #cf222e">-0.32</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v6** | top_k_x2 | 58.33 | 67.74<br><sub style="color: #cf222e">-0.32</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v6** | all | 58.33 | 67.74<br><sub style="color: #cf222e">-0.32</sub> | 54.33<br><sub style="color: #cf222e">-1.34</sub> |
| **v7** | top_k | 59.03<br><sub style="color: #2da44e">+0.70</sub> | 68.39<br><sub style="color: #2da44e">+0.33</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v7** | top_k_x2 | 57.64<br><sub style="color: #cf222e">-0.69</sub> | 68.71<br><sub style="color: #2da44e">+0.65</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v7** | all | 59.72<br><sub style="color: #2da44e">+1.39</sub> | 68.39<br><sub style="color: #2da44e">+0.33</sub> | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v8** | top_k | 59.72<br><sub style="color: #2da44e">+1.39</sub> | 67.10<br><sub style="color: #cf222e">-0.96</sub> | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v8** | top_k_x2 | 59.03<br><sub style="color: #2da44e">+0.70</sub> | 66.77<br><sub style="color: #cf222e">-1.29</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v8** | all | 59.03<br><sub style="color: #2da44e">+0.70</sub> | 67.42<br><sub style="color: #cf222e">-0.64</sub> | 54.00<br><sub style="color: #cf222e">-1.67</sub> |
| **v9** | top_k | 59.72<br><sub style="color: #2da44e">+1.39</sub> | 67.42<br><sub style="color: #cf222e">-0.64</sub> | 54.33<br><sub style="color: #cf222e">-1.34</sub> |
| **v9** | top_k_x2 | 59.03<br><sub style="color: #2da44e">+0.70</sub> | 67.10<br><sub style="color: #cf222e">-0.96</sub> | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v9** | all | 59.03<br><sub style="color: #2da44e">+0.70</sub> | 68.06 | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v10** | top_k | 59.72<br><sub style="color: #2da44e">+1.39</sub> | 67.42<br><sub style="color: #cf222e">-0.64</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v10** | top_k_x2 | 58.33 | 67.74<br><sub style="color: #cf222e">-0.32</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v10** | all | 59.03<br><sub style="color: #2da44e">+0.70</sub> | 67.74<br><sub style="color: #cf222e">-0.32</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |

### Chemistry · Standard

| Bias | Loss | col_chem | hs_chem | all_part |
| :--- | :--- | :--- | :--- | :--- |
| **BEFORE** | — | 45.00 | 44.83 | 55.67 |
| **v4** | top_k | 46.00<br><sub style="color: #2da44e">+1.00</sub> | 43.84<br><sub style="color: #cf222e">-0.99</sub> | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v4** | top_k_x2 | 47.00<br><sub style="color: #2da44e">+2.00</sub> | 44.83 | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v4** | all | 47.00<br><sub style="color: #2da44e">+2.00</sub> | 44.83 | 56.00<br><sub style="color: #2da44e">+0.33</sub> |
| **v5** | top_k | 52.00<br><sub style="color: #2da44e">+7.00</sub> | 44.33<br><sub style="color: #cf222e">-0.50</sub> | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v5** | top_k_x2 | 51.00<br><sub style="color: #2da44e">+6.00</sub> | 44.83 | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v5** | all | 48.00<br><sub style="color: #2da44e">+3.00</sub> | 44.33<br><sub style="color: #cf222e">-0.50</sub> | 53.33<br><sub style="color: #cf222e">-2.34</sub> |
| **v6** | top_k | 47.00<br><sub style="color: #2da44e">+2.00</sub> | 43.84<br><sub style="color: #cf222e">-0.99</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v6** | top_k_x2 | 48.00<br><sub style="color: #2da44e">+3.00</sub> | 44.33<br><sub style="color: #cf222e">-0.50</sub> | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v6** | all | 46.00<br><sub style="color: #2da44e">+1.00</sub> | 45.32<br><sub style="color: #2da44e">+0.49</sub> | 55.67 |
| **v7** | top_k | 47.00<br><sub style="color: #2da44e">+2.00</sub> | 44.83 | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v7** | top_k_x2 | 46.00<br><sub style="color: #2da44e">+1.00</sub> | 45.32<br><sub style="color: #2da44e">+0.49</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v7** | all | 45.00 | 44.33<br><sub style="color: #cf222e">-0.50</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v8** | top_k | 47.00<br><sub style="color: #2da44e">+2.00</sub> | 44.83 | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v8** | top_k_x2 | 49.00<br><sub style="color: #2da44e">+4.00</sub> | 45.81<br><sub style="color: #2da44e">+0.98</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v8** | all | 46.00<br><sub style="color: #2da44e">+1.00</sub> | 44.83 | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v9** | top_k | 48.00<br><sub style="color: #2da44e">+3.00</sub> | 44.33<br><sub style="color: #cf222e">-0.50</sub> | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v9** | top_k_x2 | 46.00<br><sub style="color: #2da44e">+1.00</sub> | 43.35<br><sub style="color: #cf222e">-1.48</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v9** | all | 48.00<br><sub style="color: #2da44e">+3.00</sub> | 46.31<br><sub style="color: #2da44e">+1.48</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v10** | top_k | 47.00<br><sub style="color: #2da44e">+2.00</sub> | 43.84<br><sub style="color: #cf222e">-0.99</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v10** | top_k_x2 | 48.00<br><sub style="color: #2da44e">+3.00</sub> | 44.83 | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v10** | all | 48.00<br><sub style="color: #2da44e">+3.00</sub> | 44.33<br><sub style="color: #cf222e">-0.50</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |

***

### Math · Corrected

| Bias | Loss | col_math | abstr_alg | elem_math | hs_math | all_part |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BEFORE** | — | 35.00 | 38.00 | 40.21 | 30.37 | 55.67 |
| **v4** | top_k | 40.00<br><sub style="color: #2da44e">+5.00</sub> | 39.00<br><sub style="color: #2da44e">+1.00</sub> | 40.74<br><sub style="color: #2da44e">+0.53</sub> | 32.22<br><sub style="color: #2da44e">+1.85</sub> | 55.67 |
| **v4** | top_k_x2 | 38.00<br><sub style="color: #2da44e">+3.00</sub> | 39.00<br><sub style="color: #2da44e">+1.00</sub> | 41.53<br><sub style="color: #2da44e">+1.32</sub> | 31.85<br><sub style="color: #2da44e">+1.48</sub> | 55.67 |
| **v4** | all | 40.00<br><sub style="color: #2da44e">+5.00</sub> | 39.00<br><sub style="color: #2da44e">+1.00</sub> | 40.74<br><sub style="color: #2da44e">+0.53</sub> | 32.96<br><sub style="color: #2da44e">+2.59</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v5** | top_k | 31.00<br><sub style="color: #cf222e">-4.00</sub> | 35.00<br><sub style="color: #cf222e">-3.00</sub> | 34.66<br><sub style="color: #cf222e">-5.55</sub> | 31.85<br><sub style="color: #2da44e">+1.48</sub> | 54.00<br><sub style="color: #cf222e">-1.67</sub> |
| **v5** | top_k_x2 | 35.00 | 39.00<br><sub style="color: #2da44e">+1.00</sub> | 32.54<br><sub style="color: #cf222e">-7.67</sub> | 31.11<br><sub style="color: #2da44e">+0.74</sub> | 52.67<br><sub style="color: #cf222e">-3.00</sub> |
| **v5** | all | 36.00<br><sub style="color: #2da44e">+1.00</sub> | 35.00<br><sub style="color: #cf222e">-3.00</sub> | 37.04<br><sub style="color: #cf222e">-3.17</sub> | 32.22<br><sub style="color: #2da44e">+1.85</sub> | 51.00<br><sub style="color: #cf222e">-4.67</sub> |
| **v6** | top_k | 39.00<br><sub style="color: #2da44e">+4.00</sub> | 40.00<br><sub style="color: #2da44e">+2.00</sub> | 41.01<br><sub style="color: #2da44e">+0.80</sub> | 31.48<br><sub style="color: #2da44e">+1.11</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v6** | top_k_x2 | 39.00<br><sub style="color: #2da44e">+4.00</sub> | 39.00<br><sub style="color: #2da44e">+1.00</sub> | 41.80<br><sub style="color: #2da44e">+1.59</sub> | 32.22<br><sub style="color: #2da44e">+1.85</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v6** | all | 40.00<br><sub style="color: #2da44e">+5.00</sub> | 39.00<br><sub style="color: #2da44e">+1.00</sub> | 41.53<br><sub style="color: #2da44e">+1.32</sub> | 31.11<br><sub style="color: #2da44e">+0.74</sub> | 55.67 |
| **v7** | top_k | 38.00<br><sub style="color: #2da44e">+3.00</sub> | 40.00<br><sub style="color: #2da44e">+2.00</sub> | 39.95<br><sub style="color: #cf222e">-0.26</sub> | 31.11<br><sub style="color: #2da44e">+0.74</sub> | 55.67 |
| **v7** | top_k_x2 | 37.00<br><sub style="color: #2da44e">+2.00</sub> | 40.00<br><sub style="color: #2da44e">+2.00</sub> | 39.95<br><sub style="color: #cf222e">-0.26</sub> | 30.37 | 56.00<br><sub style="color: #2da44e">+0.33</sub> |
| **v7** | all | 37.00<br><sub style="color: #2da44e">+2.00</sub> | 40.00<br><sub style="color: #2da44e">+2.00</sub> | 41.01<br><sub style="color: #2da44e">+0.80</sub> | 30.00<br><sub style="color: #cf222e">-0.37</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v8** | top_k | 39.00<br><sub style="color: #2da44e">+4.00</sub> | 39.00<br><sub style="color: #2da44e">+1.00</sub> | 40.74<br><sub style="color: #2da44e">+0.53</sub> | 31.85<br><sub style="color: #2da44e">+1.48</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v8** | top_k_x2 | 37.00<br><sub style="color: #2da44e">+2.00</sub> | 39.00<br><sub style="color: #2da44e">+1.00</sub> | 40.74<br><sub style="color: #2da44e">+0.53</sub> | 32.22<br><sub style="color: #2da44e">+1.85</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v8** | all | 40.00<br><sub style="color: #2da44e">+5.00</sub> | 40.00<br><sub style="color: #2da44e">+2.00</sub> | 41.27<br><sub style="color: #2da44e">+1.06</sub> | 31.48<br><sub style="color: #2da44e">+1.11</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v9** | top_k | 40.00<br><sub style="color: #2da44e">+5.00</sub> | 39.00<br><sub style="color: #2da44e">+1.00</sub> | 40.48<br><sub style="color: #2da44e">+0.27</sub> | 31.85<br><sub style="color: #2da44e">+1.48</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v9** | top_k_x2 | 37.00<br><sub style="color: #2da44e">+2.00</sub> | 40.00<br><sub style="color: #2da44e">+2.00</sub> | 40.21 | 30.37 | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v9** | all | 38.00<br><sub style="color: #2da44e">+3.00</sub> | 39.00<br><sub style="color: #2da44e">+1.00</sub> | 40.48<br><sub style="color: #2da44e">+0.27</sub> | 32.22<br><sub style="color: #2da44e">+1.85</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v10** | top_k | 39.00<br><sub style="color: #2da44e">+4.00</sub> | 39.00<br><sub style="color: #2da44e">+1.00</sub> | 41.27<br><sub style="color: #2da44e">+1.06</sub> | 30.37 | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v10** | top_k_x2 | 38.00<br><sub style="color: #2da44e">+3.00</sub> | 38.00 | 41.53<br><sub style="color: #2da44e">+1.32</sub> | 31.48<br><sub style="color: #2da44e">+1.11</sub> | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v10** | all | 40.00<br><sub style="color: #2da44e">+5.00</sub> | 39.00<br><sub style="color: #2da44e">+1.00</sub> | 40.48<br><sub style="color: #2da44e">+0.27</sub> | 31.11<br><sub style="color: #2da44e">+0.74</sub> | 56.00<br><sub style="color: #2da44e">+0.33</sub> |

### Physics · Corrected

| Bias | Loss | col_phys | conc_phys | hs_phys | all_part |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BEFORE** | — | 32.35 | 51.91 | 37.09 | 55.67 |
| **v4** | top_k | 32.35 | 50.64<br><sub style="color: #cf222e">-1.27</sub> | 36.42<br><sub style="color: #cf222e">-0.67</sub> | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v4** | top_k_x2 | 35.29<br><sub style="color: #2da44e">+2.94</sub> | 51.49<br><sub style="color: #cf222e">-0.42</sub> | 37.75<br><sub style="color: #2da44e">+0.66</sub> | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v4** | all | 35.29<br><sub style="color: #2da44e">+2.94</sub> | 50.64<br><sub style="color: #cf222e">-1.27</sub> | 36.42<br><sub style="color: #cf222e">-0.67</sub> | 56.00<br><sub style="color: #2da44e">+0.33</sub> |
| **v5** | top_k | 39.22<br><sub style="color: #2da44e">+6.87</sub> | 47.23<br><sub style="color: #cf222e">-4.68</sub> | 32.45<br><sub style="color: #cf222e">-4.64</sub> | 52.00<br><sub style="color: #cf222e">-3.67</sub> |
| **v5** | top_k_x2 | 38.24<br><sub style="color: #2da44e">+5.89</sub> | 47.66<br><sub style="color: #cf222e">-4.25</sub> | 33.77<br><sub style="color: #cf222e">-3.32</sub> | 51.67<br><sub style="color: #cf222e">-4.00</sub> |
| **v5** | all | 37.25<br><sub style="color: #2da44e">+4.90</sub> | 45.11<br><sub style="color: #cf222e">-6.80</sub> | 32.45<br><sub style="color: #cf222e">-4.64</sub> | 52.67<br><sub style="color: #cf222e">-3.00</sub> |
| **v6** | top_k | 34.31<br><sub style="color: #2da44e">+1.96</sub> | 51.06<br><sub style="color: #cf222e">-0.85</sub> | 36.42<br><sub style="color: #cf222e">-0.67</sub> | 54.33<br><sub style="color: #cf222e">-1.34</sub> |
| **v6** | top_k_x2 | 34.31<br><sub style="color: #2da44e">+1.96</sub> | 51.06<br><sub style="color: #cf222e">-0.85</sub> | 36.42<br><sub style="color: #cf222e">-0.67</sub> | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v6** | all | 35.29<br><sub style="color: #2da44e">+2.94</sub> | 50.64<br><sub style="color: #cf222e">-1.27</sub> | 37.75<br><sub style="color: #2da44e">+0.66</sub> | 56.00<br><sub style="color: #2da44e">+0.33</sub> |
| **v7** | top_k | 34.31<br><sub style="color: #2da44e">+1.96</sub> | 51.49<br><sub style="color: #cf222e">-0.42</sub> | 36.42<br><sub style="color: #cf222e">-0.67</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v7** | top_k_x2 | 34.31<br><sub style="color: #2da44e">+1.96</sub> | 51.06<br><sub style="color: #cf222e">-0.85</sub> | 37.09 | 56.00<br><sub style="color: #2da44e">+0.33</sub> |
| **v7** | all | 34.31<br><sub style="color: #2da44e">+1.96</sub> | 50.64<br><sub style="color: #cf222e">-1.27</sub> | 35.76<br><sub style="color: #cf222e">-1.33</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v8** | top_k | 33.33<br><sub style="color: #2da44e">+0.98</sub> | 50.64<br><sub style="color: #cf222e">-1.27</sub> | 35.10<br><sub style="color: #cf222e">-1.99</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v8** | top_k_x2 | 33.33<br><sub style="color: #2da44e">+0.98</sub> | 51.06<br><sub style="color: #cf222e">-0.85</sub> | 37.09 | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v8** | all | 32.35 | 50.64<br><sub style="color: #cf222e">-1.27</sub> | 39.07<br><sub style="color: #2da44e">+1.98</sub> | 56.00<br><sub style="color: #2da44e">+0.33</sub> |
| **v9** | top_k | 33.33<br><sub style="color: #2da44e">+0.98</sub> | 50.64<br><sub style="color: #cf222e">-1.27</sub> | 34.44<br><sub style="color: #cf222e">-2.65</sub> | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v9** | top_k_x2 | 33.33<br><sub style="color: #2da44e">+0.98</sub> | 50.64<br><sub style="color: #cf222e">-1.27</sub> | 35.10<br><sub style="color: #cf222e">-1.99</sub> | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v9** | all | 36.27<br><sub style="color: #2da44e">+3.92</sub> | 50.64<br><sub style="color: #cf222e">-1.27</sub> | 35.76<br><sub style="color: #cf222e">-1.33</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v10** | top_k | 33.33<br><sub style="color: #2da44e">+0.98</sub> | 50.64<br><sub style="color: #cf222e">-1.27</sub> | 35.76<br><sub style="color: #cf222e">-1.33</sub> | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v10** | top_k_x2 | 35.29<br><sub style="color: #2da44e">+2.94</sub> | 51.06<br><sub style="color: #cf222e">-0.85</sub> | 37.09 | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v10** | all | 35.29<br><sub style="color: #2da44e">+2.94</sub> | 50.21<br><sub style="color: #cf222e">-1.70</sub> | 36.42<br><sub style="color: #cf222e">-0.67</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |

### History · Corrected

| Bias | Loss | hs_world | hs_euro | hs_us | prehistory | all_part |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BEFORE** | — | 70.89 | 68.48 | 69.12 | 65.43 | 55.67 |
| **v4** | top_k | 71.73<br><sub style="color: #2da44e">+0.84</sub> | 68.48 | 69.61<br><sub style="color: #2da44e">+0.49</sub> | 64.81<br><sub style="color: #cf222e">-0.62</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v4** | top_k_x2 | 71.73<br><sub style="color: #2da44e">+0.84</sub> | 68.48 | 70.10<br><sub style="color: #2da44e">+0.98</sub> | 64.81<br><sub style="color: #cf222e">-0.62</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v4** | all | 72.15<br><sub style="color: #2da44e">+1.26</sub> | 67.88<br><sub style="color: #cf222e">-0.60</sub> | 71.57<br><sub style="color: #2da44e">+2.45</sub> | 65.12<br><sub style="color: #cf222e">-0.31</sub> | 55.67 |
| **v5** | top_k | 75.53<br><sub style="color: #2da44e">+4.64</sub> | 68.48 | 72.06<br><sub style="color: #2da44e">+2.94</sub> | 66.67<br><sub style="color: #2da44e">+1.24</sub> | 58.67<br><sub style="color: #2da44e">+3.00</sub> |
| **v5** | top_k_x2 | 74.26<br><sub style="color: #2da44e">+3.37</sub> | 67.27<br><sub style="color: #cf222e">-1.21</sub> | 73.53<br><sub style="color: #2da44e">+4.41</sub> | 66.67<br><sub style="color: #2da44e">+1.24</sub> | 57.33<br><sub style="color: #2da44e">+1.66</sub> |
| **v5** | all | 74.26<br><sub style="color: #2da44e">+3.37</sub> | 68.48 | 73.53<br><sub style="color: #2da44e">+4.41</sub> | 66.98<br><sub style="color: #2da44e">+1.55</sub> | 58.33<br><sub style="color: #2da44e">+2.66</sub> |
| **v6** | top_k | 71.73<br><sub style="color: #2da44e">+0.84</sub> | 68.48 | 70.10<br><sub style="color: #2da44e">+0.98</sub> | 64.81<br><sub style="color: #cf222e">-0.62</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v6** | top_k_x2 | 72.15<br><sub style="color: #2da44e">+1.26</sub> | 68.48 | 70.59<br><sub style="color: #2da44e">+1.47</sub> | 65.12<br><sub style="color: #cf222e">-0.31</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v6** | all | 72.15<br><sub style="color: #2da44e">+1.26</sub> | 67.88<br><sub style="color: #cf222e">-0.60</sub> | 71.57<br><sub style="color: #2da44e">+2.45</sub> | 65.12<br><sub style="color: #cf222e">-0.31</sub> | 56.00<br><sub style="color: #2da44e">+0.33</sub> |
| **v7** | top_k | 71.73<br><sub style="color: #2da44e">+0.84</sub> | 69.09<br><sub style="color: #2da44e">+0.61</sub> | 69.61<br><sub style="color: #2da44e">+0.49</sub> | 64.81<br><sub style="color: #cf222e">-0.62</sub> | 56.00<br><sub style="color: #2da44e">+0.33</sub> |
| **v7** | top_k_x2 | 71.31<br><sub style="color: #2da44e">+0.42</sub> | 69.09<br><sub style="color: #2da44e">+0.61</sub> | 69.61<br><sub style="color: #2da44e">+0.49</sub> | 65.12<br><sub style="color: #cf222e">-0.31</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v7** | all | 71.31<br><sub style="color: #2da44e">+0.42</sub> | 69.09<br><sub style="color: #2da44e">+0.61</sub> | 69.61<br><sub style="color: #2da44e">+0.49</sub> | 64.81<br><sub style="color: #cf222e">-0.62</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v8** | top_k | 72.15<br><sub style="color: #2da44e">+1.26</sub> | 68.48 | 71.08<br><sub style="color: #2da44e">+1.96</sub> | 64.81<br><sub style="color: #cf222e">-0.62</sub> | 55.67 |
| **v8** | top_k_x2 | 71.73<br><sub style="color: #2da44e">+0.84</sub> | 67.88<br><sub style="color: #cf222e">-0.60</sub> | 70.59<br><sub style="color: #2da44e">+1.47</sub> | 64.20<br><sub style="color: #cf222e">-1.23</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v8** | all | 71.31<br><sub style="color: #2da44e">+0.42</sub> | 67.88<br><sub style="color: #cf222e">-0.60</sub> | 71.08<br><sub style="color: #2da44e">+1.96</sub> | 63.89<br><sub style="color: #cf222e">-1.54</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v9** | top_k | 71.73<br><sub style="color: #2da44e">+0.84</sub> | 68.48 | 70.59<br><sub style="color: #2da44e">+1.47</sub> | 64.81<br><sub style="color: #cf222e">-0.62</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v9** | top_k_x2 | 72.15<br><sub style="color: #2da44e">+1.26</sub> | 68.48 | 70.10<br><sub style="color: #2da44e">+0.98</sub> | 63.89<br><sub style="color: #cf222e">-1.54</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v9** | all | 72.15<br><sub style="color: #2da44e">+1.26</sub> | 68.48 | 70.10<br><sub style="color: #2da44e">+0.98</sub> | 64.81<br><sub style="color: #cf222e">-0.62</sub> | 56.00<br><sub style="color: #2da44e">+0.33</sub> |
| **v10** | top_k | 72.15<br><sub style="color: #2da44e">+1.26</sub> | 68.48 | 69.61<br><sub style="color: #2da44e">+0.49</sub> | 64.51<br><sub style="color: #cf222e">-0.92</sub> | 55.67 |
| **v10** | top_k_x2 | 70.89 | 68.48 | 69.61<br><sub style="color: #2da44e">+0.49</sub> | 64.81<br><sub style="color: #cf222e">-0.62</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v10** | all | 70.04<br><sub style="color: #cf222e">-0.85</sub> | 67.88<br><sub style="color: #cf222e">-0.60</sub> | 70.10<br><sub style="color: #2da44e">+0.98</sub> | 64.81<br><sub style="color: #cf222e">-0.62</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |

### Biology · Corrected

| Bias | Loss | col_bio | hs_bio | all_part |
| :--- | :--- | :--- | :--- | :--- |
| **BEFORE** | — | 58.33 | 68.06 | 55.67 |
| **v4** | top_k | 59.03<br><sub style="color: #2da44e">+0.70</sub> | 67.10<br><sub style="color: #cf222e">-0.96</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v4** | top_k_x2 | 58.33 | 68.06 | 54.33<br><sub style="color: #cf222e">-1.34</sub> |
| **v4** | all | 59.03<br><sub style="color: #2da44e">+0.70</sub> | 68.06 | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v5** | top_k | 55.56<br><sub style="color: #cf222e">-2.77</sub> | 67.10<br><sub style="color: #cf222e">-0.96</sub> | 53.67<br><sub style="color: #cf222e">-2.00</sub> |
| **v5** | top_k_x2 | 54.86<br><sub style="color: #cf222e">-3.47</sub> | 66.45<br><sub style="color: #cf222e">-1.61</sub> | 53.67<br><sub style="color: #cf222e">-2.00</sub> |
| **v5** | all | 52.78<br><sub style="color: #cf222e">-5.55</sub> | 64.52<br><sub style="color: #cf222e">-3.54</sub> | 54.00<br><sub style="color: #cf222e">-1.67</sub> |
| **v6** | top_k | 59.03<br><sub style="color: #2da44e">+0.70</sub> | 67.42<br><sub style="color: #cf222e">-0.64</sub> | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v6** | top_k_x2 | 57.64<br><sub style="color: #cf222e">-0.69</sub> | 68.06 | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v6** | all | 59.72<br><sub style="color: #2da44e">+1.39</sub> | 67.74<br><sub style="color: #cf222e">-0.32</sub> | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v7** | top_k | 58.33 | 68.06 | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v7** | top_k_x2 | 58.33 | 68.39<br><sub style="color: #2da44e">+0.33</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v7** | all | 59.03<br><sub style="color: #2da44e">+0.70</sub> | 68.06 | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v8** | top_k | 59.72<br><sub style="color: #2da44e">+1.39</sub> | 68.06 | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v8** | top_k_x2 | 58.33 | 67.42<br><sub style="color: #cf222e">-0.64</sub> | 54.33<br><sub style="color: #cf222e">-1.34</sub> |
| **v8** | all | 56.94<br><sub style="color: #cf222e">-1.39</sub> | 67.42<br><sub style="color: #cf222e">-0.64</sub> | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v9** | top_k | 59.72<br><sub style="color: #2da44e">+1.39</sub> | 67.74<br><sub style="color: #cf222e">-0.32</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v9** | top_k_x2 | 59.03<br><sub style="color: #2da44e">+0.70</sub> | 68.06 | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v9** | all | 58.33 | 68.06 | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v10** | top_k | 59.03<br><sub style="color: #2da44e">+0.70</sub> | 67.42<br><sub style="color: #cf222e">-0.64</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v10** | top_k_x2 | 58.33 | 67.42<br><sub style="color: #cf222e">-0.64</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v10** | all | 58.33 | 68.06 | 55.33<br><sub style="color: #cf222e">-0.34</sub> |

### Chemistry · Corrected

| Bias | Loss | col_chem | hs_chem | all_part |
| :--- | :--- | :--- | :--- | :--- |
| **BEFORE** | — | 45.00 | 44.83 | 55.67 |
| **v4** | top_k | 47.00<br><sub style="color: #2da44e">+2.00</sub> | 44.83 | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v4** | top_k_x2 | 48.00<br><sub style="color: #2da44e">+3.00</sub> | 45.32<br><sub style="color: #2da44e">+0.49</sub> | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v4** | all | 46.00<br><sub style="color: #2da44e">+1.00</sub> | 45.81<br><sub style="color: #2da44e">+0.98</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v5** | top_k | 56.00<br><sub style="color: #2da44e">+11.00</sub> | 44.33<br><sub style="color: #cf222e">-0.50</sub> | 54.33<br><sub style="color: #cf222e">-1.34</sub> |
| **v5** | top_k_x2 | 59.00<br><sub style="color: #2da44e">+14.00</sub> | 44.83 | 53.33<br><sub style="color: #cf222e">-2.34</sub> |
| **v5** | all | 57.00<br><sub style="color: #2da44e">+12.00</sub> | 43.35<br><sub style="color: #cf222e">-1.48</sub> | 52.33<br><sub style="color: #cf222e">-3.34</sub> |
| **v6** | top_k | 48.00<br><sub style="color: #2da44e">+3.00</sub> | 44.33<br><sub style="color: #cf222e">-0.50</sub> | 55.67 |
| **v6** | top_k_x2 | 48.00<br><sub style="color: #2da44e">+3.00</sub> | 45.32<br><sub style="color: #2da44e">+0.49</sub> | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v6** | all | 48.00<br><sub style="color: #2da44e">+3.00</sub> | 45.81<br><sub style="color: #2da44e">+0.98</sub> | 55.67 |
| **v7** | top_k | 46.00<br><sub style="color: #2da44e">+1.00</sub> | 44.83 | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v7** | top_k_x2 | 46.00<br><sub style="color: #2da44e">+1.00</sub> | 46.31<br><sub style="color: #2da44e">+1.48</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v7** | all | 48.00<br><sub style="color: #2da44e">+3.00</sub> | 46.31<br><sub style="color: #2da44e">+1.48</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v8** | top_k | 48.00<br><sub style="color: #2da44e">+3.00</sub> | 44.33<br><sub style="color: #cf222e">-0.50</sub> | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v8** | top_k_x2 | 49.00<br><sub style="color: #2da44e">+4.00</sub> | 45.81<br><sub style="color: #2da44e">+0.98</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v8** | all | 46.00<br><sub style="color: #2da44e">+1.00</sub> | 44.83 | 55.33<br><sub style="color: #cf222e">-0.34</sub> |
| **v9** | top_k | 48.00<br><sub style="color: #2da44e">+3.00</sub> | 44.33<br><sub style="color: #cf222e">-0.50</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v9** | top_k_x2 | 48.00<br><sub style="color: #2da44e">+3.00</sub> | 44.33<br><sub style="color: #cf222e">-0.50</sub> | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v9** | all | 47.00<br><sub style="color: #2da44e">+2.00</sub> | 44.83 | 55.00<br><sub style="color: #cf222e">-0.67</sub> |
| **v10** | top_k | 47.00<br><sub style="color: #2da44e">+2.00</sub> | 44.83 | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v10** | top_k_x2 | 48.00<br><sub style="color: #2da44e">+3.00</sub> | 44.83 | 54.67<br><sub style="color: #cf222e">-1.00</sub> |
| **v10** | all | 48.00<br><sub style="color: #2da44e">+3.00</sub> | 45.32<br><sub style="color: #2da44e">+0.49</sub> | 55.33<br><sub style="color: #cf222e">-0.34</sub> |


## Qwen-3

### Math · Standard

| Bias | Loss | col_math | abstr_alg | elem_math | hs_math | all_part |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BEFORE** | — | 73.00 | 73.00 | 85.19 | 65.93 | 85.00 |
| **v4** | top_k | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 85.45<br><sub style="color: #2da44e">+0.26</sub> | 65.56<br><sub style="color: #cf222e">-0.37</sub> | 85.00 |
| **v4** | top_k_x2 | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 85.71<br><sub style="color: #2da44e">+0.52</sub> | 65.93 | 85.00 |
| **v4** | all | 73.00 | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 85.19 | 65.93 | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v5** | top_k | 70.00<br><sub style="color: #cf222e">-3.00</sub> | 71.00<br><sub style="color: #cf222e">-2.00</sub> | 84.66<br><sub style="color: #cf222e">-0.53</sub> | 62.59<br><sub style="color: #cf222e">-3.34</sub> | 80.33<br><sub style="color: #cf222e">-4.67</sub> |
| **v5** | top_k_x2 | 69.00<br><sub style="color: #cf222e">-4.00</sub> | 67.00<br><sub style="color: #cf222e">-6.00</sub> | 83.60<br><sub style="color: #cf222e">-1.59</sub> | 61.11<br><sub style="color: #cf222e">-4.82</sub> | 80.67<br><sub style="color: #cf222e">-4.33</sub> |
| **v5** | all | 69.00<br><sub style="color: #cf222e">-4.00</sub> | 68.00<br><sub style="color: #cf222e">-5.00</sub> | 85.19 | 62.22<br><sub style="color: #cf222e">-3.71</sub> | 80.67<br><sub style="color: #cf222e">-4.33</sub> |
| **v6** | top_k | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 85.45<br><sub style="color: #2da44e">+0.26</sub> | 65.93 | 85.00 |
| **v6** | top_k_x2 | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 85.19 | 65.56<br><sub style="color: #cf222e">-0.37</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v6** | all | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 84.92<br><sub style="color: #cf222e">-0.27</sub> | 65.56<br><sub style="color: #cf222e">-0.37</sub> | 85.00 |
| **v7** | top_k | 72.00<br><sub style="color: #cf222e">-1.00</sub> | 73.00 | 85.45<br><sub style="color: #2da44e">+0.26</sub> | 65.56<br><sub style="color: #cf222e">-0.37</sub> | 84.33<br><sub style="color: #cf222e">-0.67</sub> |
| **v7** | top_k_x2 | 72.00<br><sub style="color: #cf222e">-1.00</sub> | 73.00 | 85.19 | 65.56<br><sub style="color: #cf222e">-0.37</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v7** | all | 73.00 | 73.00 | 85.19 | 66.30<br><sub style="color: #2da44e">+0.37</sub> | 85.00 |
| **v8** | top_k | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 73.00 | 85.45<br><sub style="color: #2da44e">+0.26</sub> | 65.93 | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v8** | top_k_x2 | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 85.45<br><sub style="color: #2da44e">+0.26</sub> | 66.30<br><sub style="color: #2da44e">+0.37</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v8** | all | 75.00<br><sub style="color: #2da44e">+2.00</sub> | 73.00 | 85.71<br><sub style="color: #2da44e">+0.52</sub> | 65.56<br><sub style="color: #cf222e">-0.37</sub> | 85.00 |
| **v9** | top_k | 73.00 | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 85.19 | 65.19<br><sub style="color: #cf222e">-0.74</sub> | 85.00 |
| **v9** | top_k_x2 | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 85.45<br><sub style="color: #2da44e">+0.26</sub> | 65.19<br><sub style="color: #cf222e">-0.74</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v9** | all | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 85.45<br><sub style="color: #2da44e">+0.26</sub> | 65.56<br><sub style="color: #cf222e">-0.37</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v10** | top_k | 73.00 | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 85.71<br><sub style="color: #2da44e">+0.52</sub> | 65.56<br><sub style="color: #cf222e">-0.37</sub> | 85.00 |
| **v10** | top_k_x2 | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 85.45<br><sub style="color: #2da44e">+0.26</sub> | 65.93 | 85.00 |
| **v10** | all | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 85.71<br><sub style="color: #2da44e">+0.52</sub> | 65.56<br><sub style="color: #cf222e">-0.37</sub> | 85.00 |

### Physics · Standard

| Bias | Loss | col_phys | conc_phys | hs_phys | all_part |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BEFORE** | — | 71.57 | 89.79 | 72.85 | 85.00 |
| **v4** | top_k | 69.61<br><sub style="color: #cf222e">-1.96</sub> | 89.79 | 72.85 | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v4** | top_k_x2 | 70.59<br><sub style="color: #cf222e">-0.98</sub> | 89.36<br><sub style="color: #cf222e">-0.43</sub> | 73.51<br><sub style="color: #2da44e">+0.66</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v4** | all | 69.61<br><sub style="color: #cf222e">-1.96</sub> | 89.79 | 73.51<br><sub style="color: #2da44e">+0.66</sub> | 85.00 |
| **v5** | top_k | 63.73<br><sub style="color: #cf222e">-7.84</sub> | 88.09<br><sub style="color: #cf222e">-1.70</sub> | 76.16<br><sub style="color: #2da44e">+3.31</sub> | 80.67<br><sub style="color: #cf222e">-4.33</sub> |
| **v5** | top_k_x2 | 62.75<br><sub style="color: #cf222e">-8.82</sub> | 88.09<br><sub style="color: #cf222e">-1.70</sub> | 74.83<br><sub style="color: #2da44e">+1.98</sub> | 80.00<br><sub style="color: #cf222e">-5.00</sub> |
| **v5** | all | 62.75<br><sub style="color: #cf222e">-8.82</sub> | 88.09<br><sub style="color: #cf222e">-1.70</sub> | 73.51<br><sub style="color: #2da44e">+0.66</sub> | 81.00<br><sub style="color: #cf222e">-4.00</sub> |
| **v6** | top_k | 68.63<br><sub style="color: #cf222e">-2.94</sub> | 89.79 | 72.85 | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v6** | top_k_x2 | 71.57 | 89.79 | 72.19<br><sub style="color: #cf222e">-0.66</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v6** | all | 68.63<br><sub style="color: #cf222e">-2.94</sub> | 90.21<br><sub style="color: #2da44e">+0.42</sub> | 72.19<br><sub style="color: #cf222e">-0.66</sub> | 85.00 |
| **v7** | top_k | 71.57 | 88.94<br><sub style="color: #cf222e">-0.85</sub> | 72.19<br><sub style="color: #cf222e">-0.66</sub> | 85.00 |
| **v7** | top_k_x2 | 71.57 | 89.79 | 72.19<br><sub style="color: #cf222e">-0.66</sub> | 85.00 |
| **v7** | all | 70.59<br><sub style="color: #cf222e">-0.98</sub> | 89.36<br><sub style="color: #cf222e">-0.43</sub> | 72.19<br><sub style="color: #cf222e">-0.66</sub> | 85.00 |
| **v8** | top_k | 70.59<br><sub style="color: #cf222e">-0.98</sub> | 89.79 | 72.85 | 85.33<br><sub style="color: #2da44e">+0.33</sub> |
| **v8** | top_k_x2 | 70.59<br><sub style="color: #cf222e">-0.98</sub> | 89.36<br><sub style="color: #cf222e">-0.43</sub> | 72.85 | 84.33<br><sub style="color: #cf222e">-0.67</sub> |
| **v8** | all | 70.59<br><sub style="color: #cf222e">-0.98</sub> | 90.21<br><sub style="color: #2da44e">+0.42</sub> | 72.85 | 84.33<br><sub style="color: #cf222e">-0.67</sub> |
| **v9** | top_k | 71.57 | 89.36<br><sub style="color: #cf222e">-0.43</sub> | 72.85 | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v9** | top_k_x2 | 68.63<br><sub style="color: #cf222e">-2.94</sub> | 89.36<br><sub style="color: #cf222e">-0.43</sub> | 72.19<br><sub style="color: #cf222e">-0.66</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v9** | all | 70.59<br><sub style="color: #cf222e">-0.98</sub> | 90.21<br><sub style="color: #2da44e">+0.42</sub> | 72.19<br><sub style="color: #cf222e">-0.66</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v10** | top_k | 67.65<br><sub style="color: #cf222e">-3.92</sub> | 89.79 | 73.51<br><sub style="color: #2da44e">+0.66</sub> | 84.33<br><sub style="color: #cf222e">-0.67</sub> |
| **v10** | top_k_x2 | 69.61<br><sub style="color: #cf222e">-1.96</sub> | 90.21<br><sub style="color: #2da44e">+0.42</sub> | 72.85 | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v10** | all | 67.65<br><sub style="color: #cf222e">-3.92</sub> | 89.79 | 72.85 | 84.67<br><sub style="color: #cf222e">-0.33</sub> |

### History · Standard

| Bias | Loss | hs_world | hs_euro | hs_us | prehistory | all_part |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BEFORE** | — | 90.72 | 85.45 | 91.18 | 88.27 | 85.00 |
| **v4** | top_k | 91.14<br><sub style="color: #2da44e">+0.42</sub> | 85.45 | 91.18 | 87.35<br><sub style="color: #cf222e">-0.92</sub> | 84.00<br><sub style="color: #cf222e">-1.00</sub> |
| **v4** | top_k_x2 | 91.14<br><sub style="color: #2da44e">+0.42</sub> | 85.45 | 91.18 | 87.65<br><sub style="color: #cf222e">-0.62</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v4** | all | 90.30<br><sub style="color: #cf222e">-0.42</sub> | 85.45 | 91.67<br><sub style="color: #2da44e">+0.49</sub> | 87.35<br><sub style="color: #cf222e">-0.92</sub> | 85.33<br><sub style="color: #2da44e">+0.33</sub> |
| **v5** | top_k | 90.30<br><sub style="color: #cf222e">-0.42</sub> | 83.64<br><sub style="color: #cf222e">-1.81</sub> | 90.69<br><sub style="color: #cf222e">-0.49</sub> | 87.35<br><sub style="color: #cf222e">-0.92</sub> | 82.33<br><sub style="color: #cf222e">-2.67</sub> |
| **v5** | top_k_x2 | 89.45<br><sub style="color: #cf222e">-1.27</sub> | 82.42<br><sub style="color: #cf222e">-3.03</sub> | 89.71<br><sub style="color: #cf222e">-1.47</sub> | 87.65<br><sub style="color: #cf222e">-0.62</sub> | 82.00<br><sub style="color: #cf222e">-3.00</sub> |
| **v5** | all | 90.30<br><sub style="color: #cf222e">-0.42</sub> | 83.03<br><sub style="color: #cf222e">-2.42</sub> | 89.22<br><sub style="color: #cf222e">-1.96</sub> | 87.96<br><sub style="color: #cf222e">-0.31</sub> | 79.67<br><sub style="color: #cf222e">-5.33</sub> |
| **v6** | top_k | 90.72 | 85.45 | 91.67<br><sub style="color: #2da44e">+0.49</sub> | 87.35<br><sub style="color: #cf222e">-0.92</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v6** | top_k_x2 | 91.14<br><sub style="color: #2da44e">+0.42</sub> | 85.45 | 91.67<br><sub style="color: #2da44e">+0.49</sub> | 87.65<br><sub style="color: #cf222e">-0.62</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v6** | all | 91.14<br><sub style="color: #2da44e">+0.42</sub> | 85.45 | 91.67<br><sub style="color: #2da44e">+0.49</sub> | 87.96<br><sub style="color: #cf222e">-0.31</sub> | 85.33<br><sub style="color: #2da44e">+0.33</sub> |
| **v7** | top_k | 90.30<br><sub style="color: #cf222e">-0.42</sub> | 86.06<br><sub style="color: #2da44e">+0.61</sub> | 91.67<br><sub style="color: #2da44e">+0.49</sub> | 87.65<br><sub style="color: #cf222e">-0.62</sub> | 85.00 |
| **v7** | top_k_x2 | 90.30<br><sub style="color: #cf222e">-0.42</sub> | 86.06<br><sub style="color: #2da44e">+0.61</sub> | 91.67<br><sub style="color: #2da44e">+0.49</sub> | 87.65<br><sub style="color: #cf222e">-0.62</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v7** | all | 90.30<br><sub style="color: #cf222e">-0.42</sub> | 85.45 | 91.67<br><sub style="color: #2da44e">+0.49</sub> | 87.96<br><sub style="color: #cf222e">-0.31</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v8** | top_k | 91.14<br><sub style="color: #2da44e">+0.42</sub> | 85.45 | 91.18 | 87.65<br><sub style="color: #cf222e">-0.62</sub> | 85.00 |
| **v8** | top_k_x2 | 91.14<br><sub style="color: #2da44e">+0.42</sub> | 84.85<br><sub style="color: #cf222e">-0.60</sub> | 91.18 | 87.35<br><sub style="color: #cf222e">-0.92</sub> | 84.33<br><sub style="color: #cf222e">-0.67</sub> |
| **v8** | all | 89.87<br><sub style="color: #cf222e">-0.85</sub> | 85.45 | 91.18 | 87.65<br><sub style="color: #cf222e">-0.62</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v9** | top_k | 91.14<br><sub style="color: #2da44e">+0.42</sub> | 85.45 | 91.18 | 87.65<br><sub style="color: #cf222e">-0.62</sub> | 85.00 |
| **v9** | top_k_x2 | 90.30<br><sub style="color: #cf222e">-0.42</sub> | 84.85<br><sub style="color: #cf222e">-0.60</sub> | 91.67<br><sub style="color: #2da44e">+0.49</sub> | 87.96<br><sub style="color: #cf222e">-0.31</sub> | 85.00 |
| **v9** | all | 90.72 | 85.45 | 91.18 | 87.65<br><sub style="color: #cf222e">-0.62</sub> | 85.00 |
| **v10** | top_k | 90.72 | 85.45 | 91.67<br><sub style="color: #2da44e">+0.49</sub> | 87.35<br><sub style="color: #cf222e">-0.92</sub> | 84.33<br><sub style="color: #cf222e">-0.67</sub> |
| **v10** | top_k_x2 | 90.72 | 85.45 | 91.67<br><sub style="color: #2da44e">+0.49</sub> | 87.65<br><sub style="color: #cf222e">-0.62</sub> | 84.33<br><sub style="color: #cf222e">-0.67</sub> |
| **v10** | all | 90.30<br><sub style="color: #cf222e">-0.42</sub> | 85.45 | 91.67<br><sub style="color: #2da44e">+0.49</sub> | 87.65<br><sub style="color: #cf222e">-0.62</sub> | 85.00 |

### Biology · Standard

| Bias | Loss | col_bio | hs_bio | all_part |
| :--- | :--- | :--- | :--- | :--- |
| **BEFORE** | — | 93.75 | 93.23 | 85.00 |
| **v4** | top_k | 94.44<br><sub style="color: #2da44e">+0.69</sub> | 94.52<br><sub style="color: #2da44e">+1.29</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v4** | top_k_x2 | 94.44<br><sub style="color: #2da44e">+0.69</sub> | 94.19<br><sub style="color: #2da44e">+0.96</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v4** | all | 94.44<br><sub style="color: #2da44e">+0.69</sub> | 93.55<br><sub style="color: #2da44e">+0.32</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v5** | top_k | 93.75 | 92.90<br><sub style="color: #cf222e">-0.33</sub> | 78.00<br><sub style="color: #cf222e">-7.00</sub> |
| **v5** | top_k_x2 | 94.44<br><sub style="color: #2da44e">+0.69</sub> | 92.90<br><sub style="color: #cf222e">-0.33</sub> | 77.67<br><sub style="color: #cf222e">-7.33</sub> |
| **v5** | all | 94.44<br><sub style="color: #2da44e">+0.69</sub> | 92.58<br><sub style="color: #cf222e">-0.65</sub> | 77.33<br><sub style="color: #cf222e">-7.67</sub> |
| **v6** | top_k | 93.75 | 93.55<br><sub style="color: #2da44e">+0.32</sub> | 85.00 |
| **v6** | top_k_x2 | 94.44<br><sub style="color: #2da44e">+0.69</sub> | 93.23 | 85.00 |
| **v6** | all | 93.75 | 93.55<br><sub style="color: #2da44e">+0.32</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v7** | top_k | 93.75 | 94.19<br><sub style="color: #2da44e">+0.96</sub> | 85.00 |
| **v7** | top_k_x2 | 94.44<br><sub style="color: #2da44e">+0.69</sub> | 93.55<br><sub style="color: #2da44e">+0.32</sub> | 85.00 |
| **v7** | all | 93.75 | 93.23 | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v8** | top_k | 94.44<br><sub style="color: #2da44e">+0.69</sub> | 93.87<br><sub style="color: #2da44e">+0.64</sub> | 84.33<br><sub style="color: #cf222e">-0.67</sub> |
| **v8** | top_k_x2 | 93.75 | 94.52<br><sub style="color: #2da44e">+1.29</sub> | 84.00<br><sub style="color: #cf222e">-1.00</sub> |
| **v8** | all | 94.44<br><sub style="color: #2da44e">+0.69</sub> | 93.55<br><sub style="color: #2da44e">+0.32</sub> | 85.00 |
| **v9** | top_k | 93.75 | 93.87<br><sub style="color: #2da44e">+0.64</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v9** | top_k_x2 | 94.44<br><sub style="color: #2da44e">+0.69</sub> | 93.55<br><sub style="color: #2da44e">+0.32</sub> | 85.00 |
| **v9** | all | 94.44<br><sub style="color: #2da44e">+0.69</sub> | 93.87<br><sub style="color: #2da44e">+0.64</sub> | 85.00 |
| **v10** | top_k | 93.75 | 93.87<br><sub style="color: #2da44e">+0.64</sub> | 85.00 |
| **v10** | top_k_x2 | 93.75 | 93.87<br><sub style="color: #2da44e">+0.64</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v10** | all | 94.44<br><sub style="color: #2da44e">+0.69</sub> | 93.55<br><sub style="color: #2da44e">+0.32</sub> | 84.33<br><sub style="color: #cf222e">-0.67</sub> |

### Chemistry · Standard

| Bias | Loss | col_chem | hs_chem | all_part |
| :--- | :--- | :--- | :--- | :--- |
| **BEFORE** | — | 61.00 | 81.28 | 85.00 |
| **v4** | top_k | 60.00<br><sub style="color: #cf222e">-1.00</sub> | 80.30<br><sub style="color: #cf222e">-0.98</sub> | 84.33<br><sub style="color: #cf222e">-0.67</sub> |
| **v4** | top_k_x2 | 60.00<br><sub style="color: #cf222e">-1.00</sub> | 81.28 | 84.33<br><sub style="color: #cf222e">-0.67</sub> |
| **v4** | all | 60.00<br><sub style="color: #cf222e">-1.00</sub> | 80.79<br><sub style="color: #cf222e">-0.49</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v5** | top_k | 55.00<br><sub style="color: #cf222e">-6.00</sub> | 78.33<br><sub style="color: #cf222e">-2.95</sub> | 80.33<br><sub style="color: #cf222e">-4.67</sub> |
| **v5** | top_k_x2 | 58.00<br><sub style="color: #cf222e">-3.00</sub> | 77.83<br><sub style="color: #cf222e">-3.45</sub> | 81.33<br><sub style="color: #cf222e">-3.67</sub> |
| **v5** | all | 57.00<br><sub style="color: #cf222e">-4.00</sub> | 78.82<br><sub style="color: #cf222e">-2.46</sub> | 80.67<br><sub style="color: #cf222e">-4.33</sub> |
| **v6** | top_k | 60.00<br><sub style="color: #cf222e">-1.00</sub> | 82.27<br><sub style="color: #2da44e">+0.99</sub> | 85.33<br><sub style="color: #2da44e">+0.33</sub> |
| **v6** | top_k_x2 | 59.00<br><sub style="color: #cf222e">-2.00</sub> | 81.28 | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v6** | all | 61.00 | 80.30<br><sub style="color: #cf222e">-0.98</sub> | 85.00 |
| **v7** | top_k | 60.00<br><sub style="color: #cf222e">-1.00</sub> | 80.79<br><sub style="color: #cf222e">-0.49</sub> | 84.33<br><sub style="color: #cf222e">-0.67</sub> |
| **v7** | top_k_x2 | 60.00<br><sub style="color: #cf222e">-1.00</sub> | 81.28 | 85.00 |
| **v7** | all | 61.00 | 80.30<br><sub style="color: #cf222e">-0.98</sub> | 85.00 |
| **v8** | top_k | 61.00 | 80.30<br><sub style="color: #cf222e">-0.98</sub> | 84.33<br><sub style="color: #cf222e">-0.67</sub> |
| **v8** | top_k_x2 | 59.00<br><sub style="color: #cf222e">-2.00</sub> | 80.30<br><sub style="color: #cf222e">-0.98</sub> | 84.33<br><sub style="color: #cf222e">-0.67</sub> |
| **v8** | all | 58.00<br><sub style="color: #cf222e">-3.00</sub> | 80.79<br><sub style="color: #cf222e">-0.49</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v9** | top_k | 60.00<br><sub style="color: #cf222e">-1.00</sub> | 80.79<br><sub style="color: #cf222e">-0.49</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v9** | top_k_x2 | 60.00<br><sub style="color: #cf222e">-1.00</sub> | 80.30<br><sub style="color: #cf222e">-0.98</sub> | 85.00 |
| **v9** | all | 60.00<br><sub style="color: #cf222e">-1.00</sub> | 80.79<br><sub style="color: #cf222e">-0.49</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v10** | top_k | 59.00<br><sub style="color: #cf222e">-2.00</sub> | 80.30<br><sub style="color: #cf222e">-0.98</sub> | 84.33<br><sub style="color: #cf222e">-0.67</sub> |
| **v10** | top_k_x2 | 60.00<br><sub style="color: #cf222e">-1.00</sub> | 80.79<br><sub style="color: #cf222e">-0.49</sub> | 84.00<br><sub style="color: #cf222e">-1.00</sub> |
| **v10** | all | 59.00<br><sub style="color: #cf222e">-2.00</sub> | 81.28 | 85.00 |

***

### Math · Corrected

| Bias | Loss | col_math | abstr_alg | elem_math | hs_math | all_part |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BEFORE** | — | 73.00 | 73.00 | 85.19 | 65.93 | 85.00 |
| **v4** | top_k | 73.00 | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 85.45<br><sub style="color: #2da44e">+0.26</sub> | 65.56<br><sub style="color: #cf222e">-0.37</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v4** | top_k_x2 | 71.00<br><sub style="color: #cf222e">-2.00</sub> | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 85.71<br><sub style="color: #2da44e">+0.52</sub> | 65.56<br><sub style="color: #cf222e">-0.37</sub> | 85.00 |
| **v4** | all | 72.00<br><sub style="color: #cf222e">-1.00</sub> | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 85.19 | 65.56<br><sub style="color: #cf222e">-0.37</sub> | 85.00 |
| **v5** | top_k | 66.00<br><sub style="color: #cf222e">-7.00</sub> | 69.00<br><sub style="color: #cf222e">-4.00</sub> | 84.92<br><sub style="color: #cf222e">-0.27</sub> | 57.78<br><sub style="color: #cf222e">-8.15</sub> | 81.67<br><sub style="color: #cf222e">-3.33</sub> |
| **v5** | top_k_x2 | 63.00<br><sub style="color: #cf222e">-10.00</sub> | 67.00<br><sub style="color: #cf222e">-6.00</sub> | 84.13<br><sub style="color: #cf222e">-1.06</sub> | 58.52<br><sub style="color: #cf222e">-7.41</sub> | 80.33<br><sub style="color: #cf222e">-4.67</sub> |
| **v5** | all | 68.00<br><sub style="color: #cf222e">-5.00</sub> | 66.00<br><sub style="color: #cf222e">-7.00</sub> | 84.13<br><sub style="color: #cf222e">-1.06</sub> | 57.41<br><sub style="color: #cf222e">-8.52</sub> | 80.67<br><sub style="color: #cf222e">-4.33</sub> |
| **v6** | top_k | 73.00 | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 85.19 | 66.67<br><sub style="color: #2da44e">+0.74</sub> | 85.00 |
| **v6** | top_k_x2 | 73.00 | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 85.19 | 65.93 | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v6** | all | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 85.19 | 66.30<br><sub style="color: #2da44e">+0.37</sub> | 85.00 |
| **v7** | top_k | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 85.19 | 65.56<br><sub style="color: #cf222e">-0.37</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v7** | top_k_x2 | 73.00 | 73.00 | 85.45<br><sub style="color: #2da44e">+0.26</sub> | 65.56<br><sub style="color: #cf222e">-0.37</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v7** | all | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 73.00 | 85.45<br><sub style="color: #2da44e">+0.26</sub> | 65.93 | 85.00 |
| **v8** | top_k | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 85.19 | 65.56<br><sub style="color: #cf222e">-0.37</sub> | 84.33<br><sub style="color: #cf222e">-0.67</sub> |
| **v8** | top_k_x2 | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 85.71<br><sub style="color: #2da44e">+0.52</sub> | 65.56<br><sub style="color: #cf222e">-0.37</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v8** | all | 73.00 | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 84.92<br><sub style="color: #cf222e">-0.27</sub> | 65.93 | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v9** | top_k | 73.00 | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 85.19 | 65.93 | 84.33<br><sub style="color: #cf222e">-0.67</sub> |
| **v9** | top_k_x2 | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 85.71<br><sub style="color: #2da44e">+0.52</sub> | 65.56<br><sub style="color: #cf222e">-0.37</sub> | 84.33<br><sub style="color: #cf222e">-0.67</sub> |
| **v9** | all | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 85.71<br><sub style="color: #2da44e">+0.52</sub> | 66.30<br><sub style="color: #2da44e">+0.37</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v10** | top_k | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 85.45<br><sub style="color: #2da44e">+0.26</sub> | 65.56<br><sub style="color: #cf222e">-0.37</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v10** | top_k_x2 | 73.00 | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 85.71<br><sub style="color: #2da44e">+0.52</sub> | 66.30<br><sub style="color: #2da44e">+0.37</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v10** | all | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 74.00<br><sub style="color: #2da44e">+1.00</sub> | 85.45<br><sub style="color: #2da44e">+0.26</sub> | 66.30<br><sub style="color: #2da44e">+0.37</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |

### Physics · Corrected

| Bias | Loss | col_phys | conc_phys | hs_phys | all_part |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BEFORE** | — | 71.57 | 89.79 | 72.85 | 85.00 |
| **v4** | top_k | 69.61<br><sub style="color: #cf222e">-1.96</sub> | 89.79 | 72.19<br><sub style="color: #cf222e">-0.66</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v4** | top_k_x2 | 70.59<br><sub style="color: #cf222e">-0.98</sub> | 90.21<br><sub style="color: #2da44e">+0.42</sub> | 73.51<br><sub style="color: #2da44e">+0.66</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v4** | all | 69.61<br><sub style="color: #cf222e">-1.96</sub> | 89.36<br><sub style="color: #cf222e">-0.43</sub> | 74.17<br><sub style="color: #2da44e">+1.32</sub> | 84.00<br><sub style="color: #cf222e">-1.00</sub> |
| **v5** | top_k | 61.76<br><sub style="color: #cf222e">-9.81</sub> | 88.51<br><sub style="color: #cf222e">-1.28</sub> | 77.48<br><sub style="color: #2da44e">+4.63</sub> | 81.67<br><sub style="color: #cf222e">-3.33</sub> |
| **v5** | top_k_x2 | 61.76<br><sub style="color: #cf222e">-9.81</sub> | 87.66<br><sub style="color: #cf222e">-2.13</sub> | 75.50<br><sub style="color: #2da44e">+2.65</sub> | 82.00<br><sub style="color: #cf222e">-3.00</sub> |
| **v5** | all | 65.69<br><sub style="color: #cf222e">-5.88</sub> | 88.51<br><sub style="color: #cf222e">-1.28</sub> | 73.51<br><sub style="color: #2da44e">+0.66</sub> | 81.00<br><sub style="color: #cf222e">-4.00</sub> |
| **v6** | top_k | 70.59<br><sub style="color: #cf222e">-0.98</sub> | 89.79 | 72.85 | 84.33<br><sub style="color: #cf222e">-0.67</sub> |
| **v6** | top_k_x2 | 70.59<br><sub style="color: #cf222e">-0.98</sub> | 89.79 | 72.85 | 85.00 |
| **v6** | all | 71.57 | 90.21<br><sub style="color: #2da44e">+0.42</sub> | 72.85 | 85.33<br><sub style="color: #2da44e">+0.33</sub> |
| **v7** | top_k | 70.59<br><sub style="color: #cf222e">-0.98</sub> | 89.79 | 72.19<br><sub style="color: #cf222e">-0.66</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v7** | top_k_x2 | 71.57 | 89.36<br><sub style="color: #cf222e">-0.43</sub> | 72.85 | 85.00 |
| **v7** | all | 71.57 | 90.21<br><sub style="color: #2da44e">+0.42</sub> | 71.52<br><sub style="color: #cf222e">-1.33</sub> | 85.00 |
| **v8** | top_k | 68.63<br><sub style="color: #cf222e">-2.94</sub> | 88.94<br><sub style="color: #cf222e">-0.85</sub> | 72.85 | 85.00 |
| **v8** | top_k_x2 | 69.61<br><sub style="color: #cf222e">-1.96</sub> | 89.79 | 72.85 | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v8** | all | 70.59<br><sub style="color: #cf222e">-0.98</sub> | 89.79 | 72.85 | 85.33<br><sub style="color: #2da44e">+0.33</sub> |
| **v9** | top_k | 70.59<br><sub style="color: #cf222e">-0.98</sub> | 89.79 | 72.19<br><sub style="color: #cf222e">-0.66</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v9** | top_k_x2 | 70.59<br><sub style="color: #cf222e">-0.98</sub> | 90.21<br><sub style="color: #2da44e">+0.42</sub> | 72.19<br><sub style="color: #cf222e">-0.66</sub> | 84.33<br><sub style="color: #cf222e">-0.67</sub> |
| **v9** | all | 69.61<br><sub style="color: #cf222e">-1.96</sub> | 90.21<br><sub style="color: #2da44e">+0.42</sub> | 72.85 | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v10** | top_k | 69.61<br><sub style="color: #cf222e">-1.96</sub> | 90.21<br><sub style="color: #2da44e">+0.42</sub> | 72.19<br><sub style="color: #cf222e">-0.66</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v10** | top_k_x2 | 69.61<br><sub style="color: #cf222e">-1.96</sub> | 89.79 | 72.19<br><sub style="color: #cf222e">-0.66</sub> | 85.00 |
| **v10** | all | 70.59<br><sub style="color: #cf222e">-0.98</sub> | 89.79 | 72.85 | 84.67<br><sub style="color: #cf222e">-0.33</sub> |

### History · Corrected

| Bias | Loss | hs_world | hs_euro | hs_us | prehistory | all_part |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BEFORE** | — | 90.72 | 85.45 | 91.18 | 88.27 | 85.00 |
| **v4** | top_k | 90.72 | 84.85<br><sub style="color: #cf222e">-0.60</sub> | 91.18 | 87.35<br><sub style="color: #cf222e">-0.92</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v4** | top_k_x2 | 90.30<br><sub style="color: #cf222e">-0.42</sub> | 86.06<br><sub style="color: #2da44e">+0.61</sub> | 91.18 | 87.04<br><sub style="color: #cf222e">-1.23</sub> | 85.00 |
| **v4** | all | 89.87<br><sub style="color: #cf222e">-0.85</sub> | 86.06<br><sub style="color: #2da44e">+0.61</sub> | 91.67<br><sub style="color: #2da44e">+0.49</sub> | 87.96<br><sub style="color: #cf222e">-0.31</sub> | 85.00 |
| **v5** | top_k | 90.72 | 83.03<br><sub style="color: #cf222e">-2.42</sub> | 87.25<br><sub style="color: #cf222e">-3.93</sub> | 87.35<br><sub style="color: #cf222e">-0.92</sub> | 81.33<br><sub style="color: #cf222e">-3.67</sub> |
| **v5** | top_k_x2 | 88.19<br><sub style="color: #cf222e">-2.53</sub> | 84.24<br><sub style="color: #cf222e">-1.21</sub> | 90.20<br><sub style="color: #cf222e">-0.98</sub> | 86.73<br><sub style="color: #cf222e">-1.54</sub> | 82.67<br><sub style="color: #cf222e">-2.33</sub> |
| **v5** | all | 88.61<br><sub style="color: #cf222e">-2.11</sub> | 83.64<br><sub style="color: #cf222e">-1.81</sub> | 87.75<br><sub style="color: #cf222e">-3.43</sub> | 87.04<br><sub style="color: #cf222e">-1.23</sub> | 81.33<br><sub style="color: #cf222e">-3.67</sub> |
| **v6** | top_k | 90.72 | 86.06<br><sub style="color: #2da44e">+0.61</sub> | 91.18 | 87.65<br><sub style="color: #cf222e">-0.62</sub> | 85.00 |
| **v6** | top_k_x2 | 90.72 | 86.06<br><sub style="color: #2da44e">+0.61</sub> | 90.69<br><sub style="color: #cf222e">-0.49</sub> | 87.35<br><sub style="color: #cf222e">-0.92</sub> | 85.00 |
| **v6** | all | 90.72 | 86.06<br><sub style="color: #2da44e">+0.61</sub> | 91.18 | 87.35<br><sub style="color: #cf222e">-0.92</sub> | 85.00 |
| **v7** | top_k | 90.30<br><sub style="color: #cf222e">-0.42</sub> | 85.45 | 91.18 | 87.65<br><sub style="color: #cf222e">-0.62</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v7** | top_k_x2 | 90.72 | 85.45 | 91.18 | 87.65<br><sub style="color: #cf222e">-0.62</sub> | 85.00 |
| **v7** | all | 90.30<br><sub style="color: #cf222e">-0.42</sub> | 86.06<br><sub style="color: #2da44e">+0.61</sub> | 91.67<br><sub style="color: #2da44e">+0.49</sub> | 87.65<br><sub style="color: #cf222e">-0.62</sub> | 85.00 |
| **v8** | top_k | 90.72 | 85.45 | 91.18 | 87.35<br><sub style="color: #cf222e">-0.92</sub> | 85.00 |
| **v8** | top_k_x2 | 90.30<br><sub style="color: #cf222e">-0.42</sub> | 85.45 | 91.18 | 87.04<br><sub style="color: #cf222e">-1.23</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v8** | all | 90.72 | 84.85<br><sub style="color: #cf222e">-0.60</sub> | 91.18 | 87.65<br><sub style="color: #cf222e">-0.62</sub> | 84.33<br><sub style="color: #cf222e">-0.67</sub> |
| **v9** | top_k | 90.30<br><sub style="color: #cf222e">-0.42</sub> | 85.45 | 91.18 | 87.65<br><sub style="color: #cf222e">-0.62</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v9** | top_k_x2 | 91.14<br><sub style="color: #2da44e">+0.42</sub> | 85.45 | 91.18 | 87.65<br><sub style="color: #cf222e">-0.62</sub> | 85.00 |
| **v9** | all | 91.14<br><sub style="color: #2da44e">+0.42</sub> | 85.45 | 91.18 | 87.35<br><sub style="color: #cf222e">-0.92</sub> | 85.00 |
| **v10** | top_k | 91.14<br><sub style="color: #2da44e">+0.42</sub> | 84.85<br><sub style="color: #cf222e">-0.60</sub> | 91.18 | 87.35<br><sub style="color: #cf222e">-0.92</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v10** | top_k_x2 | 91.14<br><sub style="color: #2da44e">+0.42</sub> | 86.06<br><sub style="color: #2da44e">+0.61</sub> | 91.18 | 87.35<br><sub style="color: #cf222e">-0.92</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v10** | all | 90.72 | 86.06<br><sub style="color: #2da44e">+0.61</sub> | 90.69<br><sub style="color: #cf222e">-0.49</sub> | 87.35<br><sub style="color: #cf222e">-0.92</sub> | 85.00 |

### Biology · Corrected

| Bias | Loss | col_bio | hs_bio | all_part |
| :--- | :--- | :--- | :--- | :--- |
| **BEFORE** | — | 93.75 | 93.23 | 85.00 |
| **v4** | top_k | 93.06<br><sub style="color: #cf222e">-0.69</sub> | 93.55<br><sub style="color: #2da44e">+0.32</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v4** | top_k_x2 | 93.06<br><sub style="color: #cf222e">-0.69</sub> | 93.55<br><sub style="color: #2da44e">+0.32</sub> | 83.67<br><sub style="color: #cf222e">-1.33</sub> |
| **v4** | all | 93.75 | 94.19<br><sub style="color: #2da44e">+0.96</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v5** | top_k | 93.75 | 92.26<br><sub style="color: #cf222e">-0.97</sub> | 79.00<br><sub style="color: #cf222e">-6.00</sub> |
| **v5** | top_k_x2 | 93.06<br><sub style="color: #cf222e">-0.69</sub> | 91.61<br><sub style="color: #cf222e">-1.62</sub> | 77.67<br><sub style="color: #cf222e">-7.33</sub> |
| **v5** | all | 94.44<br><sub style="color: #2da44e">+0.69</sub> | 92.58<br><sub style="color: #cf222e">-0.65</sub> | 77.00<br><sub style="color: #cf222e">-8.00</sub> |
| **v6** | top_k | 94.44<br><sub style="color: #2da44e">+0.69</sub> | 93.55<br><sub style="color: #2da44e">+0.32</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v6** | top_k_x2 | 93.75 | 93.23 | 84.33<br><sub style="color: #cf222e">-0.67</sub> |
| **v6** | all | 94.44<br><sub style="color: #2da44e">+0.69</sub> | 93.55<br><sub style="color: #2da44e">+0.32</sub> | 84.33<br><sub style="color: #cf222e">-0.67</sub> |
| **v7** | top_k | 94.44<br><sub style="color: #2da44e">+0.69</sub> | 93.55<br><sub style="color: #2da44e">+0.32</sub> | 85.00 |
| **v7** | top_k_x2 | 93.75 | 93.23 | 85.00 |
| **v7** | all | 93.75 | 93.23 | 85.00 |
| **v8** | top_k | 93.06<br><sub style="color: #cf222e">-0.69</sub> | 93.55<br><sub style="color: #2da44e">+0.32</sub> | 84.00<br><sub style="color: #cf222e">-1.00</sub> |
| **v8** | top_k_x2 | 93.75 | 93.87<br><sub style="color: #2da44e">+0.64</sub> | 84.33<br><sub style="color: #cf222e">-0.67</sub> |
| **v8** | all | 93.06<br><sub style="color: #cf222e">-0.69</sub> | 93.87<br><sub style="color: #2da44e">+0.64</sub> | 84.00<br><sub style="color: #cf222e">-1.00</sub> |
| **v9** | top_k | 94.44<br><sub style="color: #2da44e">+0.69</sub> | 93.23 | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v9** | top_k_x2 | 93.75 | 93.87<br><sub style="color: #2da44e">+0.64</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v9** | all | 94.44<br><sub style="color: #2da44e">+0.69</sub> | 93.87<br><sub style="color: #2da44e">+0.64</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v10** | top_k | 93.75 | 93.55<br><sub style="color: #2da44e">+0.32</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v10** | top_k_x2 | 93.75 | 94.19<br><sub style="color: #2da44e">+0.96</sub> | 84.33<br><sub style="color: #cf222e">-0.67</sub> |
| **v10** | all | 94.44<br><sub style="color: #2da44e">+0.69</sub> | 93.23 | 84.00<br><sub style="color: #cf222e">-1.00</sub> |

### Chemistry · Corrected

| Bias | Loss | col_chem | hs_chem | all_part |
| :--- | :--- | :--- | :--- | :--- |
| **BEFORE** | — | 61.00 | 81.28 | 85.00 |
| **v4** | top_k | 60.00<br><sub style="color: #cf222e">-1.00</sub> | 81.28 | 84.33<br><sub style="color: #cf222e">-0.67</sub> |
| **v4** | top_k_x2 | 60.00<br><sub style="color: #cf222e">-1.00</sub> | 80.30<br><sub style="color: #cf222e">-0.98</sub> | 84.00<br><sub style="color: #cf222e">-1.00</sub> |
| **v4** | all | 60.00<br><sub style="color: #cf222e">-1.00</sub> | 81.77<br><sub style="color: #2da44e">+0.49</sub> | 84.33<br><sub style="color: #cf222e">-0.67</sub> |
| **v5** | top_k | 56.00<br><sub style="color: #cf222e">-5.00</sub> | 78.82<br><sub style="color: #cf222e">-2.46</sub> | 81.33<br><sub style="color: #cf222e">-3.67</sub> |
| **v5** | top_k_x2 | 56.00<br><sub style="color: #cf222e">-5.00</sub> | 79.80<br><sub style="color: #cf222e">-1.48</sub> | 81.67<br><sub style="color: #cf222e">-3.33</sub> |
| **v5** | all | 55.00<br><sub style="color: #cf222e">-6.00</sub> | 78.33<br><sub style="color: #cf222e">-2.95</sub> | 82.33<br><sub style="color: #cf222e">-2.67</sub> |
| **v6** | top_k | 61.00 | 81.28 | 84.00<br><sub style="color: #cf222e">-1.00</sub> |
| **v6** | top_k_x2 | 60.00<br><sub style="color: #cf222e">-1.00</sub> | 81.28 | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v6** | all | 60.00<br><sub style="color: #cf222e">-1.00</sub> | 81.28 | 85.00 |
| **v7** | top_k | 60.00<br><sub style="color: #cf222e">-1.00</sub> | 80.79<br><sub style="color: #cf222e">-0.49</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v7** | top_k_x2 | 60.00<br><sub style="color: #cf222e">-1.00</sub> | 80.79<br><sub style="color: #cf222e">-0.49</sub> | 85.00 |
| **v7** | all | 61.00 | 80.79<br><sub style="color: #cf222e">-0.49</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |
| **v8** | top_k | 60.00<br><sub style="color: #cf222e">-1.00</sub> | 80.79<br><sub style="color: #cf222e">-0.49</sub> | 84.33<br><sub style="color: #cf222e">-0.67</sub> |
| **v8** | top_k_x2 | 60.00<br><sub style="color: #cf222e">-1.00</sub> | 80.30<br><sub style="color: #cf222e">-0.98</sub> | 84.33<br><sub style="color: #cf222e">-0.67</sub> |
| **v8** | all | 61.00 | 81.28 | 83.67<br><sub style="color: #cf222e">-1.33</sub> |
| **v9** | top_k | 60.00<br><sub style="color: #cf222e">-1.00</sub> | 81.28 | 84.33<br><sub style="color: #cf222e">-0.67</sub> |
| **v9** | top_k_x2 | 60.00<br><sub style="color: #cf222e">-1.00</sub> | 80.79<br><sub style="color: #cf222e">-0.49</sub> | 84.33<br><sub style="color: #cf222e">-0.67</sub> |
| **v9** | all | 61.00 | 80.79<br><sub style="color: #cf222e">-0.49</sub> | 84.33<br><sub style="color: #cf222e">-0.67</sub> |
| **v10** | top_k | 61.00 | 81.28 | 84.33<br><sub style="color: #cf222e">-0.67</sub> |
| **v10** | top_k_x2 | 60.00<br><sub style="color: #cf222e">-1.00</sub> | 80.79<br><sub style="color: #cf222e">-0.49</sub> | 84.33<br><sub style="color: #cf222e">-0.67</sub> |
| **v10** | all | 60.00<br><sub style="color: #cf222e">-1.00</sub> | 80.79<br><sub style="color: #cf222e">-0.49</sub> | 84.67<br><sub style="color: #cf222e">-0.33</sub> |