[-------------------- op='LSTM', nonlinearity='None', dtype='float' ---------------------]
                                            |  1.11.0a0+git251686f  |  1.11.0a0+git9d80e6d
1 threads: -------------------------------------------------------------------------------
      x=(1125, 44, 40),   op=(40, 160, 1)   |          13.0         |          37.6       
      x=(1709, 8, 40),    op=(40, 104, 4)   |          45.9         |          15.8       
      x=(1892, 47, 104),  op=(104, 88, 4)   |          53.0         |          17.5       
      x=(2560, 3, 368),   op=(368, 40, 4)   |          65.0         |           6.1       
      x=(2091, 7, 320),   op=(320, 152, 4)  |          55.2         |          47.5       
      x=(3854, 16, 112),  op=(112, 128, 1)  |          36.6         |          10.2       
      x=(2058, 62, 16),   op=(16, 168, 1)   |          29.9         |         156.9       
      x=(5005, 8, 80),    op=(80, 128, 4)   |         130.1         |          47.0       
      x=(1314, 20, 344),  op=(344, 32, 3)   |          27.3         |           2.3       
      x=(2721, 6, 128),   op=(128, 72, 1)   |          18.6         |           2.4       
      x=(3047, 6, 104),   op=(104, 40, 1)   |          20.7         |           2.0       
      x=(5654, 5, 232),   op=(232, 64, 4)   |         146.5         |          13.5       
      x=(5895, 1, 288),   op=(288, 152, 4)  |         158.0         |         129.5       
      x=(3401, 47, 16),   op=(16, 56, 4)    |          97.7         |          13.1       
      x=(1981, 45, 56),   op=(56, 152, 4)   |          90.2         |         269.4       
      x=(5826, 1, 200),   op=(200, 160, 4)  |         154.6         |         127.8       
      x=(3095, 9, 16),    op=(16, 152, 4)   |          80.1         |          75.4       
      x=(1252, 12, 304),  op=(304, 192, 1)  |          13.8         |          17.6       
      x=(3717, 45, 24),   op=(24, 80, 3)    |          81.2         |          22.4       
      x=(1212, 55, 56),   op=(56, 112, 4)   |          53.5         |          27.0       

Times are in milliseconds (ms).

[--------------------- op='LSTM', nonlinearity='None', dtype='half' ---------------------]
                                            |  1.11.0a0+git251686f  |  1.11.0a0+git9d80e6d
1 threads: -------------------------------------------------------------------------------
      x=(1125, 44, 40),   op=(40, 160, 1)   |          16.0         |          36.9       
      x=(1709, 8, 40),    op=(40, 104, 4)   |          51.6         |          16.2       
      x=(1892, 47, 104),  op=(104, 88, 4)   |          56.4         |          14.8       
      x=(2560, 3, 368),   op=(368, 40, 4)   |          73.5         |           5.9       
      x=(2091, 7, 320),   op=(320, 152, 4)  |          61.0         |          46.9       
      x=(3854, 16, 112),  op=(112, 128, 1)  |          29.3         |          10.1       
      x=(2058, 62, 16),   op=(16, 168, 1)   |          27.6         |         154.5       
      x=(5005, 8, 80),    op=(80, 128, 4)   |         147.3         |          48.1       
      x=(1314, 20, 344),  op=(344, 32, 3)   |          29.1         |           2.0       
      x=(2721, 6, 128),   op=(128, 72, 1)   |          21.1         |           2.3       
      x=(3047, 6, 104),   op=(104, 40, 1)   |          23.3         |           2.0       
      x=(5654, 5, 232),   op=(232, 64, 4)   |         162.0         |          13.0       
      x=(5895, 1, 288),   op=(288, 152, 4)  |         160.5         |         131.7       
      x=(3401, 47, 16),   op=(16, 56, 4)    |          98.9         |          11.4       
      x=(1981, 45, 56),   op=(56, 152, 4)   |         106.7         |         261.2       
      x=(5826, 1, 200),   op=(200, 160, 4)  |         159.7         |         129.7       
      x=(3095, 9, 16),    op=(16, 152, 4)   |          90.8         |          74.8       
      x=(1252, 12, 304),  op=(304, 192, 1)  |           9.8         |          17.0       
      x=(3717, 45, 24),   op=(24, 80, 3)    |          82.9         |          20.7       
      x=(1212, 55, 56),   op=(56, 112, 4)   |          53.3         |          24.1       

Times are in milliseconds (ms).

[--------------------- op='RNN', nonlinearity='relu', dtype='float' ---------------------]
                                            |  1.11.0a0+git251686f  |  1.11.0a0+git9d80e6d
1 threads: -------------------------------------------------------------------------------
      x=(1125, 44, 40),   op=(40, 160, 1)   |          10.9         |           1.6       
      x=(4814, 6, 160),   op=(160, 360, 1)  |          44.2         |          48.9       
      x=(2058, 62, 16),   op=(16, 168, 1)   |          21.4         |           3.3       
      x=(4720, 4, 352),   op=(352, 288, 1)  |          33.2         |          27.6       
      x=(1556, 18, 32),   op=(32, 88, 2)    |          23.7         |           1.4       
      x=(1314, 20, 344),  op=(344, 32, 3)   |          28.4         |           1.4       
      x=(2391, 20, 80),   op=(80, 224, 3)   |          50.7         |           8.0       
      x=(5133, 20, 40),   op=(40, 136, 4)   |         146.3         |          14.1       
      x=(1109, 49, 144),  op=(144, 88, 4)   |          33.4         |           3.2       
      x=(3047, 6, 104),   op=(104, 296, 1)  |          21.2         |          21.0       
      x=(1406, 34, 120),  op=(120, 120, 1)  |          10.4         |           1.0       
      x=(3212, 1, 24),    op=(24, 80, 3)    |          60.3         |           3.7       
      x=(5654, 5, 232),   op=(232, 320, 4)  |         145.6         |         156.2       
      x=(4600, 16, 24),   op=(24, 264, 1)   |          42.6         |          43.6       
      x=(5895, 1, 288),   op=(288, 152, 4)  |         145.4         |          13.2       
      x=(3401, 47, 16),   op=(16, 56, 4)    |          92.9         |           5.8       
      x=(2165, 20, 104),  op=(104, 64, 4)   |          60.4         |           3.3       
      x=(5826, 1, 200),   op=(200, 160, 4)  |         144.4         |          13.2       
      x=(4936, 7, 216),   op=(216, 208, 4)  |         128.6         |          18.2       
      x=(1824, 3, 80),    op=(80, 336, 3)   |          35.2         |          45.8       

Times are in milliseconds (ms).

[--------------------- op='RNN', nonlinearity='relu', dtype='half' ----------------------]
                                            |  1.11.0a0+git251686f  |  1.11.0a0+git9d80e6d
1 threads: -------------------------------------------------------------------------------
      x=(1125, 44, 40),   op=(40, 160, 1)   |          9004.3       |          1457.8     
      x=(4814, 6, 160),   op=(160, 360, 1)  |         37091.5       |         47785.0     
      x=(2058, 62, 16),   op=(16, 168, 1)   |         15998.0       |          3096.1     
      x=(4720, 4, 352),   op=(352, 288, 1)  |         37022.1       |         27181.4     
      x=(1556, 18, 32),   op=(32, 88, 2)    |         24294.7       |          1298.4     
      x=(1314, 20, 344),  op=(344, 32, 3)   |         36097.7       |          1053.6     
      x=(2391, 20, 80),   op=(80, 224, 3)   |         59937.1       |          6429.9     
      x=(5133, 20, 40),   op=(40, 136, 4)   |        164982.9       |         12391.8     
      x=(1109, 49, 144),  op=(144, 88, 4)   |         33088.1       |          2477.5     
      x=(3047, 6, 104),   op=(104, 296, 1)  |         23442.9       |         20754.7     
      x=(1406, 34, 120),  op=(120, 120, 1)  |         11129.7       |           796.2     
      x=(3212, 1, 24),    op=(24, 80, 3)    |         62294.1       |          3701.0     
      x=(5654, 5, 232),   op=(232, 320, 4)  |        173917.9       |        155272.5     
      x=(4600, 16, 24),   op=(24, 264, 1)   |         33232.5       |         16591.9     
      x=(5895, 1, 288),   op=(288, 152, 4)  |        153507.3       |         12848.2     
      x=(3401, 47, 16),   op=(16, 56, 4)    |        100826.2       |          4919.8     
      x=(2165, 20, 104),  op=(104, 64, 4)   |         64412.9       |          2976.6     
      x=(5826, 1, 200),   op=(200, 160, 4)  |        152906.1       |         12620.5     
      x=(4936, 7, 216),   op=(216, 208, 4)  |        149831.8       |         16742.4     
      x=(1824, 3, 80),    op=(80, 336, 3)   |         40315.8       |         45714.2     

Times are in microseconds (us).

[--------------------- op='RNN', nonlinearity='tanh', dtype='float' ---------------------]
                                            |  1.11.0a0+git251686f  |  1.11.0a0+git9d80e6d
1 threads: -------------------------------------------------------------------------------
      x=(1125, 44, 40),   op=(40, 160, 1)   |          11.0         |           1.6       
      x=(4814, 6, 160),   op=(160, 360, 1)  |          44.1         |          49.7       
      x=(2058, 62, 16),   op=(16, 168, 1)   |          21.5         |           3.4       
      x=(4720, 4, 352),   op=(352, 288, 1)  |          32.2         |          27.9       
      x=(1556, 18, 32),   op=(32, 88, 2)    |          21.9         |           1.5       
      x=(1314, 20, 344),  op=(344, 32, 3)   |          27.3         |           1.4       
      x=(2391, 20, 80),   op=(80, 224, 3)   |          50.1         |           8.0       
      x=(5133, 20, 40),   op=(40, 136, 4)   |         143.1         |          14.4       
      x=(1109, 49, 144),  op=(144, 88, 4)   |          31.4         |           3.3       
      x=(3047, 6, 104),   op=(104, 296, 1)  |          20.5         |          21.0       
      x=(1406, 34, 120),  op=(120, 120, 1)  |          12.6         |           1.1       
      x=(3212, 1, 24),    op=(24, 80, 3)    |          58.9         |           4.0       
      x=(5654, 5, 232),   op=(232, 320, 4)  |         143.8         |         157.9       
      x=(4600, 16, 24),   op=(24, 264, 1)   |          42.8         |          43.6       
      x=(5895, 1, 288),   op=(288, 152, 4)  |         148.1         |          14.3       
      x=(3401, 47, 16),   op=(16, 56, 4)    |          95.2         |           6.0       
      x=(2165, 20, 104),  op=(104, 64, 4)   |          59.8         |           3.6       
      x=(5826, 1, 200),   op=(200, 160, 4)  |         141.9         |          14.0       
      x=(4936, 7, 216),   op=(216, 208, 4)  |         133.8         |          18.9       
      x=(1824, 3, 80),    op=(80, 336, 3)   |          35.0         |          46.0       

Times are in milliseconds (ms).

[--------------------- op='RNN', nonlinearity='tanh', dtype='half' ----------------------]
                                            |  1.11.0a0+git251686f  |  1.11.0a0+git9d80e6d
1 threads: -------------------------------------------------------------------------------
      x=(1125, 44, 40),   op=(40, 160, 1)   |          8699.4       |          2273.1     
      x=(4814, 6, 160),   op=(160, 360, 1)  |         36739.4       |         49041.4     
      x=(2058, 62, 16),   op=(16, 168, 1)   |         15829.2       |          3191.3     
      x=(4720, 4, 352),   op=(352, 288, 1)  |         36356.4       |         27534.9     
      x=(1556, 18, 32),   op=(32, 88, 2)    |         25412.9       |          1400.1     
      x=(1314, 20, 344),  op=(344, 32, 3)   |         29222.8       |          1170.8     
      x=(2391, 20, 80),   op=(80, 224, 3)   |         54226.1       |          6683.8     
      x=(5133, 20, 40),   op=(40, 136, 4)   |        153959.3       |         13245.0     
      x=(1109, 49, 144),  op=(144, 88, 4)   |         34216.5       |          2671.6     
      x=(3047, 6, 104),   op=(104, 296, 1)  |         25590.6       |         20831.1     
      x=(1406, 34, 120),  op=(120, 120, 1)  |         11069.6       |           878.4     
      x=(3212, 1, 24),    op=(24, 80, 3)    |         62444.6       |          4018.5     
      x=(5654, 5, 232),   op=(232, 320, 4)  |        169378.9       |        155635.3     
      x=(4600, 16, 24),   op=(24, 264, 1)   |         37354.3       |         17748.6     
      x=(5895, 1, 288),   op=(288, 152, 4)  |        152674.9       |         14047.7     
      x=(3401, 47, 16),   op=(16, 56, 4)    |        101524.1       |          5556.1     
      x=(2165, 20, 104),  op=(104, 64, 4)   |         66690.2       |          3412.7     
      x=(5826, 1, 200),   op=(200, 160, 4)  |        147741.7       |         13801.4     
      x=(4936, 7, 216),   op=(216, 208, 4)  |        147355.4       |         17528.2     
      x=(1824, 3, 80),    op=(80, 336, 3)   |         40943.4       |         45637.5     

Times are in microseconds (us).

