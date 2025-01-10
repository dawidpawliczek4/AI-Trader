# NLP 

---

<!-- pozdro dla ludzi co czytają surowego markdowna -->

<!-- TODO dodaj dobry zarys i opis modułu  -->

### sections 

1. data preprocessing
    * **stopwords removal**: removing "this", "that", "he", "it"... from sentences
    * **lemmatization**: grouping together different inflected forms of the same word eg: changing, changed -> change
    * **lowercase conversion**

2. transformation word into data
    * **bag of words**
    * **TF-IDF**
    * **embedding** (Word2Vec, GloVe, FastText)
    * **language models** (BERT, GPT)

3. model developing
    * **NN**
    * **recurrent NN**
    * **short-long term memory**
    * **knn**
    * **logistic regression**
    * **transformers**

4. model evaluation
    <!-- TODO dodaj metryki -->

5. using model for predictions 
    text processing and predictiong the sentiment 

<!-- TODO dodaj co przyjmuje główna funkcja  -->
<!-- TODO dodaj co zwraca główna funkcja  -->

<!-- 
    plan jest taki że będzie jedna główna funkcja do której przekazuje się wszystkie argumenty z sekcji które ma urzyć model do przewidzenia sentymentu. czyli zamysł jest taki że w mozna wybrać czy się chce robić jakiś data preprocessing, jak chce sie przedstawiać słowa by komputer cos z nimi zrobił ...

    w sumie to lepiej to będzie zrobić w klasie i zamiast tej głównej funkcji dodać .fit(), dzieki temu można będzie zrobić łatwo .evaluate() 
-->