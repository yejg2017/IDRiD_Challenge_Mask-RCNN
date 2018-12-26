# IDRiD_Challenge_Mask-RCNN

### The Challenge in (https://idrid.grand-challenge.org/),the the data can be download [here](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)

The Mask RCNN's application in IDRiD(only for challenge 1)

*  Step 1 : Create Dataset

First,create directory **data**
```bash
mkdir data
```
And create data
```python
python dataset.py
```
The data will store in data,they are too big,so I do not show that.

*  Step 2: Train the Model
```python 
python main.py
```
the checkpoint will be stored in directory **checkpoint**


*  Step 3: [Demo](./demo.ipynb)
Show the Mask and model evaluation
