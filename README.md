# Solution for track Traffic Vehicle Detection - SoICT Hackathon 2024

- Tổng quan giải pháp của chúng tôi như hình dưới đây:

![docs/overview.png](docs/overview.png)



__NOTE__: 
- Machine with GPU support is recommended.
- Source code was written on Window operating system and models were trained on Kaggle platform. Running on Linux may encounter errors. Please inform us for proper support and debugging.

## 0. Project Structure
- The high level structure of this project :
  - `data/`: training data are stored in this folder.
  - `test/`: public test are stored in this folder.
  - `src/infer` or `src/train`: neccessary bash scripts for training on docker container can be found in this folder. These scripts are also used for training on Kaggle platform. 
  - `saved_models/`: weights for our trained models (cus_model.pt and def_model.pt).
  - `ensemble`: this folder stores the ensemble bash script for ensembling results of my models and the "predict.txt" file as the final result (after ensembling).
  - `prediction`: The outcoming prediction for each of the 2 models (cus_predict.txt and def_predict.txt)

## 1. Environment Configuration

- After unzipping the compressed source code, run the following command to build docker image:
```bash
docker build -f Dockerfile -t dangdot .
```
- Run docker container with GPU:
```bash
docker run  --runtime=nvidia --gpus 1 -it --rm --name dangdot_container --network=host dangdot bash
```

- Run docker container without GPU:
```bash
docker run -it --rm --name dangdot_container --network=host hackathon bash
```

## 2. Model Training & Inference
### 2.1. Training
- Our two models are selected in [yolov10l, yolov11l-custom, yolov11l-default, rt-detr] after the trial experiment on two datasets (raw, synthesis).
- The final result is obtained by ensembling predictions of the 2 models.
- Run the below command to _train_ models individually and _predict_ outputs on public test. The outcoming prediction for each of models is contained in `prediction/` and weights of models are stored in `saved_models/`:  

```bash
cd /app
bash ./scripts/model5_synth_full.sh
```

- Our results reproducing will require training each model in the proposed list. One can do this sequencially by running the above command for each model, or choose another way around to train all models by running:

```bash
cd /app
bash ./scripts/train_all.sh
```

#### 2.2. Inference

- To infer all models on the private dataset, run the following command:
```bash
cd /app_
python3 model_predict.py
```

- To ensemble all models, run the following command:
```bash
cd /app
python3 predict.py
```

- `prediction.txt` lies in the folder `/app/ensemble`, to get the final output, run the following command:
```bash
cd /app/ensemble
cat prediction.txt
```