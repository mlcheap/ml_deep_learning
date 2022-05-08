# ESCO tagging - Deep Neural Net
occupation detection for this list of languages: English,Arabic,Portuguese, German, 
## Requirements
the code test on x86 system with cuda 11.3
## Models
you can use trained [models](https://drive.google.com/file/d/1yDgNpB_kGFGlbvb-RKglKCqPZ28Xl234/view?usp=sharing) and replace models directory or train your models.

## Training
the only data we use for traning is esco data or some amount of undirect labels in training Nl,Pt,De models. esco have just one related sample per class (that just define occupation). so  because we have very few amount of data compare to number of classes (about 3000) we have to use few shot learning. for this purpose we using some of best embedding model as starting point. ("all-mpnet-base-v2", "paraphrase-multilingual-mpnet-base-v2" are two embedding model).

- for training En model we use "all-mpnet-base-v2" that just support English language. we use esco data and this model as oneshot learning (for classifer we use cosine similarity classification). 

- for "Ar" we use "paraphrase-multilingual-mpnet-base-v2" model that support 50+ languages as well as Arabic. we suppose that embedding of one text is equal to meaning so meaning of one text in different language must be the same. by this way we anticipate embeds of english and arabic esco sample must be same. so by making daset of pair samples and use this idea and finetune "paraphrase-multilingual-mpnet-base-v2" by cross-encoder the result 7% increased compare to raw pretrained model. for classifer we use cosine similarity as well as En model.

- for "Pt","Nl","De" we use "paraphrase-multilingual-mpnet-base-v2" model as pretrained model but in slight differnt way. we have labeled data in different languages so we can use them to improve one langauge by this idea, and mix to idea we use in Arabic language. in first step imporoving embedding of pretrained model by make embedding of esco data in different langauge equal to english embedding. then for one lanuage as target we use labeled data + esco data of other laguage + esco data of target language as training and labeled data of target language as testing. by this method we improve accuracy for example in De about 13% 

## running flask app
backend of model is simple flask app, for running this app localy you should create python environment and install requirements python package
```sh
python3 -m venv env && source env/bin/activate  && pip install -r requirements.txt 
```
then you can run the flask app by below command
```sh
flask run 
```

## Docker

you can create contaier with running dockerfile.

By default, the Docker will expose port 5000, so change this within the
Dockerfile if necessary. When ready, simply use the Dockerfile to
build the image.

```sh
docker build -t <youruser>/occupation_detection .
```

This will create the occupation_detection image and pull in the necessary dependencies.

Once done, run the Docker image and map the port to whatever you wish on
your host. 

```sh
docker run -d --name occupation_detection --gpus all -p 5000:80  <youruser>/occupation_detection 
```

Verify the deployment by sending post and get request as api

#### Api

occupation detection
```sh
curl --location --request POST 'localhost:5000/occupations-detection' \
--data-raw '{
    "items":[{"title": "مطور تطبيقات Flutter",
 "description": "مطور تطبيقات موبايل بإستخدام Flutter، يجيد العمل ضمن الفريق ويملك حس ورغبة عالية في تطوير منتجات ذات كفاءة عالية."}],
    "model":"Ar",
    "topk":5
}'
```
output samples:

```sh
[
    "http://data.europa.eu/esco/occupation/9ba74e8a-c40c-4228-9998-eb3c7a5c11df",
    "http://data.europa.eu/esco/occupation/5c1eba1e-7820-4287-9746-4c5906320100",
    "http://data.europa.eu/esco/occupation/2eac08c2-a81a-46fc-8d75-eb0e0f3e0f6d",
    "http://data.europa.eu/esco/occupation/21af40bb-2ae3-4fc7-9b28-d2cdd8308912",
    "http://data.europa.eu/esco/occupation/0b15375e-dfdd-4047-9efb-096e0aaee7d2"
]
```

getting all exsiting models:

```sh
curl --location --request GET 'localhost:5000/all-models'
```

output samples:

```sh
[
    "EnV2",
    "En",
    "Ar",
    "De",
    "Nl",
    "Pt"
]
```
> we have two model for english language the EnV2 is the powerone laverage from dataset we generated and get better result about 85% compate to En model with about 66% accuaracy
