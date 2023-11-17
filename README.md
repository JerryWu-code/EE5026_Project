# EE5026_Project
### Guidance

The structure of the file execution is as follows: all source code is located in the "/src" folder. By running the ".py" files with the names PCA, LDA, CNN, ResNet, GMM, SVM, and draw, you can directly obtain results. Additionally, the configuration is set in the "config.py" file, which specifies the necessary parameters and paths for execution.

The output files in image format are stored in the "/figs" folder, while other types of output, such as model parameters and tables, are saved in the "data" directory.



### Folder Structure

```
.
├── LICENSE
├── README.md
├── data
│   ├── PCA_test_map.txt
│   ├── PCA_train_map.txt
│   ├── PIE
│   ├── SVM_accu_200.csv
│   ├── SVM_accu_80.csv
│   ├── cnn_log.txt
│   ├── cnn_loss.csv
│   ├── cnn_model.pth
│   ├── final_selfie
│   ├── raw_selfie
│   ├── resnet18_log.txt
│   ├── resnet18_loss.csv
│   ├── resnet18_model.pth
│   ├── test
│   └── train
├── figs
│   ├── 200_PCA_GMM_3Component.png
│   ├── 80_PCA_GMM_3Component.png
│   ├── CNN.svg
│   ├── Eigen Face #1.png
│   ├── Eigen Face #2.png
│   ├── Eigen Face #3.png
│   ├── LDA & KNN: Accuracy_Curve.png
│   ├── Mean_Face.png
│   ├── My_Selfie: Faces.png
│   ├── PCA & KNN: Accuracy_Curve.png
│   ├── PCA_Reconstru_selfie_2: Faces.png
│   ├── PCA_Reconstru_selfie_4: Faces.png
│   ├── PCA_Reconstru_selfie_7: Faces.png
│   ├── Projection of LDA.png
│   ├── Projection of PCA.png
│   ├── Training Loss History MA3.png
│   └── Training Loss History.png
├── report.pdf
├── requirements.txt
├── src
│   ├── CNN.py
│   ├── GMM.py
│   ├── KNN.py
│   ├── LDA.py
│   ├── PCA.py
│   ├── ResNet.py
│   ├── SVM.py
│   ├── __pycache__
│   ├── config.py
│   ├── data_loader.py
│   ├── draw.py
│   ├── process_selfie.py
│   └── train_test_split.py
└── task
    ├── CA2.pdf
    └── EE5907_EE5027_ Q&A for CA1.docx
```

