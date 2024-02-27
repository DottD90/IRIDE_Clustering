#!/bin/bash
python index_bursts.py "C:\Users\e.ciraci\Desktop\GIS\SNT_Bursts\snt01-bursts.shp" ".\data\Nocera_Terinese.shp" -D "C:\Users\e.ciraci\Desktop\Clustering\Nocera_Terinese\input"
python compute_aac.py "C:\Users\e.ciraci\Desktop\Clustering\Nocera_Terinese\AOIs_bursts\Nocera_Terinese.shp" --out_dir="C:\Users\e.ciraci\Desktop\Clustering\Nocera_Terinese\AAC"


python index_bursts.py "C:\Users\e.ciraci\Desktop\GIS\SNT_Bursts\snt01-bursts.shp" ".\data\Nocera_Terinese.shp" -D "C:\Users\e.ciraci\Des
ktop\TREA-GSP\SNT\S3-02-SNT-02\burst_tiles"


python index_tiles.py "C:\Users\e.ciraci\Desktop\GIS\SNT_Tiles\snt01-tiles.shp" ".\data\Nocera_Terinese.shp" -T "C:\Users\e.ciraci\Deskto
p\TREA-GSP\SNT\S3-02-SNT-04\burst_tiles"

python merge_burst.py "C:\Users\e.ciraci\Desktop\TREA-GSP\SNT\S3-02-SNT-02\AOIs_bursts\Nocera_Terinese.shp" --out_dir="C:\Users\e.ciraci\
Desktop\TREA-GSP\SNT\S3-02-SNT-02"

