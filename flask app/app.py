from flask import Flask, render_template, request
import pickle 

app = Flask(__name__)
model = pickle.load(open('Marketing_Campaign_model.pkl','rb')) # Чтение модели

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        # Получение данных с формы
        Marital_Status = int(request.form["Marital_Status"]) # Семейное положение (бинарный признак)
        Income = int(request.form["Income"]) # Доход 
        Recency = int(request.form["Recency"]) # Количество дней с последней покупки
        MntMeatProducts = int(request.form["MntMeatProducts"]) # Траты на мясные продукты 
        NumCatalogPurnaches = int(request.form["NumCatalogPurnaches"]) # Количество покупок через каталог 
        AcceptedCmp1 = int(request.form["AcceptedCmp1"]) # Принял ли предложение в 1 компании (бинарный признак)
        AcceptedCmp2 = int(request.form["AcceptedCmp2"]) # Принял ли предложение в 2 компании (бинарный признак) 
        AcceptedCmp3 = int(request.form["AcceptedCmp3"]) # Принял ли предложение в 3 компании (бинарный признак)
        AcceptedCmp4 = int(request.form["AcceptedCmp4"]) # Принял ли предложение в 4 компании (бинарный признак)
        AcceptedCmp5 = int(request.form["AcceptedCmp5"]) # Принял ли предложение в 5 компании (бинарный признак)
        Years_registered = int(request.form["Years_registered"]) # Количество лет зарегистрирован 
        Have_child = int(request.form["Have_child"]) # Есть ли дети  (бинарный признак)
        Number_childs = int(request.form["Number_childs"]) # Количество детей
        
        # Получение прогноза
        input_cols = [[Marital_Status, Income, Recency, MntMeatProducts, NumCatalogPurnaches, 
                       AcceptedCmp1, AcceptedCmp2, AcceptedCmp3, AcceptedCmp4, AcceptedCmp5, 
                       Years_registered, Have_child, Number_childs]]
        prediction = model.predict(input_cols)
        output = round(prediction[0], 2)
        return render_template("index.html", prediction_text='Прогноз: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)