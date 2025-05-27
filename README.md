
## BSCPE 3-2 Group 1: Data Mining and Data Visualization for Water Quality Prediction in Taal Lake Dashboard

See [Technical Report](https://github.com/pjpangilinan/MC02-CPEN106/blob/main/Group1_TechnicalReport_Data-Mining-and-Data-Visualization-for-Water-Quality-Prediction-in-Taal-Lake.pdf) for full documentation.

### Home

![Screenshot of the Dashboard's Homepage](https://github.com/pjpangilinan/MC02-CPEN106/blob/main/docs/home.png)

---

### Prediction

![Screenshot of the Dashboard's Selection for Model Comparison](https://github.com/pjpangilinan/MC02-CPEN106/blob/main/docs/Model%20Comparison%20Selection.png)

![Screenshot of the Dashboard's Sample Model Comparison for Surface Temp](https://github.com/pjpangilinan/MC02-CPEN106/blob/main/docs/Model%20Comparison.png)

![Screenshot of the Dashboards Selection for Time-Based Prediction + WQI Calculation](https://github.com/pjpangilinan/MC02-CPEN106/blob/main/docs/Time-Based%20Prediction%20%2B%20WQI%20Calculation%20Selection.png)

![Screenshot of the Dashboard's Sample Prediction for the Next 52 Weeks + Performance Metrics](https://github.com/pjpangilinan/MC02-CPEN106/blob/main/docs/Time-Based%20Prediction.png)

![Screenshot of the Dashboard's Sample Prediction WQI Scores + Visualization](https://github.com/pjpangilinan/MC02-CPEN106/blob/main/docs/WQI%20Calculation.png)

---

### EDA

![Dataset Overview + Heatmap](https://github.com/pjpangilinan/MC02-CPEN106/blob/main/docs/EDA%20P.1.png)

![WQI Over Time + Scatter Plot for Parameters](https://github.com/pjpangilinan/MC02-CPEN106/blob/main/docs/EDA%20P.2.png)

![Time-Series Chart](https://github.com/pjpangilinan/MC02-CPEN106/blob/main/docs/EDA%20P.3.png)

---

### Recommendations

![Recommendations for Very Poor Water Quality](https://github.com/pjpangilinan/MC02-CPEN106/blob/main/docs/R1.png)

![Other Recommendations](https://github.com/pjpangilinan/MC02-CPEN106/blob/main/docs/R2.png)

## Run Locally

Create a virtual environment

```bash
  python -m venv <name_of_venv>
```

Clone the project

```bash
  git clone https://github.com/pjpangilinan/MC02-CPEN106
```

Activate virtual environment

```bash
  # In cmd.exe
  venv\Scripts\activate.bat
  or
  Scripts/activate

  # In PowerShell
  venv\Scripts\Activate.ps1

  # In Linux/MacOS
  $ source myvenv/bin/activate
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the the streamlit website

```bash
  streamlit run app3.py
```
