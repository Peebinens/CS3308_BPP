# Task3
## 1.delete error data
```bash
python data_del.py
```
Then the correct data will be write from data.csv to data_clean.csv
## 2.run
If you want it show tqdm
```bash
python task3.py --ckp models/10-10-10.pth --config task3.yaml --device 0 --tqdm
```
If you want it render the item and container
```bash
python task3.py --ckp models/10-10-10.pth --config task3.yaml --device 0 --render
```