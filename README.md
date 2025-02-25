# Regression Models with Deep Learning

This project aims to perform regression predictions using deep learning models (MLP, RNN, CNN) on data retrieved from a MySQL database. The project includes steps for data preprocessing, model training, hyperparameter optimization, and model evaluation.

## Table of Contents

1.  [Requirements](#requirements)
2.  [Installation](#installation)
3.  [Usage](#usage)
4.  [Database Configuration](#database-configuration)
5.  [Model Selection and Improvement](#model-selection-and-improvement)
6.  [Model Saving and Loading](#model-saving-and-loading)
7.  [Notes](#notes)
8.  [License](#license)

## Requirements

* Python 3.9 or higher
* Required Python libraries:
    * `pandas`
    * `mysql-connector-python`
    * `scikit-learn`
    * `tensorflow`
    * `matplotlib`
    * `keras`
    * `keras-tuner`
    * `numpy`

    ```bash
    pip install -r requirements.txt
    ```

## Installation

1.  Install the required libraries:

    ```bash
    pip install pandas mysql-connector-python scikit-learn tensorflow matplotlib keras keras-tuner numpy
    ```

2.  Clone or download the project files to your local machine.

## Usage

1.  Configure the database connection information in the `veritabani_baglantisi_ve_veri_cekme` function according to your MySQL database.
2.  Set the `model_kayit_yolu` variable to the file path where you want to save the models.
3.  Run the project using the command `python .\data_fetching_and_preprocessing.py`.

## Database Configuration

The project uses the `mysql.connector` library to fetch data from a MySQL database. You need to set the database connection information (`host_adi`, `veritabani_adi`, `kullanici_adi`, `parola`, `tablo_adi`) in the `veritabani_baglantisi_ve_veri_cekme` function according to your database.

## Model Selection and Improvement

The project explores MLP, RNN, and CNN models. The best model is selected based on the Mean Squared Error (MSE) metric. Currently, hyperparameter optimization is performed only for the MLP model using Keras Tuner. You can add different model types or apply Keras Tuner optimization to other models as needed.

## Model Saving and Loading

Trained models are saved to the specified file path using the `model_kaydet` function. Saved models can be loaded again using the `tf.keras.models.load_model(model_kayit_yolu)` function and used for predictions on new data.

## Notes

* For RNN and CNN models, the input data should be in the form of 3D tensors (number of samples, time steps, number of features). In this code, the data is simply reshaped (e.g., considering each data point as 1 time step) to fit these models. To better capture time series features, you can divide the data into time steps and improve model performance by using more suitable RNN/CNN architectures.
* Keras Tuner optimization searches for hyperparameters within the specified ranges. You can further improve results by using a more comprehensive search or different optimization algorithms (e.g., Bayesian optimization).
* Visualizing feature importance in deep learning models may require more complex methods (SHAP values, LIME, etc.) or approaches that examine model weights/activations. The feature importance graph drawing step is omitted in this code example, but separate research and development can be done on this topic.

## License

This project is licensed under the MIT License.