import csv
import json
import os
import glob
import pandas as pd
import numpy as np
from gensim import corpora, models
import lsa_models
import pickle

def get_path_info():
    paths = json.loads(open("./SETTINGS.json").read())
    for key in paths:
        paths[key] = os.path.expandvars(paths[key])
    return paths

def identity(x):
    return x

# For pandas >= 10.1 this will trigger the columns to be parsed as strings
converters = { "FullDescription" : identity
             , "Title": identity
             , "LocationRaw": identity
             , "LocationNormalized": identity
             }

def build_data_path(dataset_name):
    path_info = get_path_info()
    return path_info["data_path"] + "/" + dataset_name + ".csv"


def get_train_df():
    train_path = build_data_path(get_path_info()["training_set_name"])
    #return pd.read_csv(train_path, converters=converters)
    return pd.read_csv(train_path)


def get_valid_df():
    valid_path = build_data_path(get_path_info()["validation_set_name"])
    # return pd.read_csv(valid_path, converters=converters)
    return pd.read_csv(valid_path)


def get_stopwords_list():
  ''' Create a list of stop words from the stop words file and return it
  '''
  stopwords_file = get_path_info()["stopwords_path"]
  stoplist = [x.rstrip("\n") for x in open(stopwords_file,"r").readlines()]
  # print stoplist
  return stoplist


def build_model_path(fieldname, extension):
    path_info = get_path_info()
    return path_info["lsa_model_path"] + "/" + path_info["training_set_name"] \
                                   + "/" + fieldname + "." + extension


def build_dictionary_path(fieldname):
    return build_model_path(fieldname, "dictionary.pickle")


def build_tfidf_path(fieldname):
    return build_model_path(fieldname, "tfidf.pickle")


def build_lsi_path(fieldname):
    return build_model_path(fieldname, "lsi.pickle")


def save(path, model):
    print "Saving model to " + path
    pickle.dump(model, open(path, "w"))


def load(path):
    print "Loading model from " + path
    if not os.path.exists(path):
      print "Load of model unsuccessful, building new one"
      return None
    return pickle.load(open(path))

def delete_models(fieldname):
    files = glob.glob(build_model_path(fieldname, "*"))
    for f in files:
       print "Removing files" + f
       os.remove(f)

def save_features(fieldname, feature_set, feature_array):
    path = build_model_path(fieldname,feature_set+".features.csv")
    print "Saving features for" + fieldname + " in " + feature_set + " to file " + path
    np.savetxt(path, feature_array, delimiter=",")


def save_model(model):
    out_path = get_paths()["model_path"]
    pickle.dump(model, open(out_path, "w"))


def load_model():
    in_path = get_paths()["model_path"]
    return pickle.load(open(in_path))


def write_submission(predictions):
    prediction_path = get_paths()["prediction_path"]
    writer = csv.writer(open(prediction_path, "w"), lineterminator="\n")
    valid = get_valid_df()
    rows = [x for x in zip(valid["Id"], predictions.flatten())]
    writer.writerow(("Id", "SalaryNormalized"))
    writer.writerows(rows)


def main():
  ''' Main is used for unit testing
  '''
  fieldname = "FullDescription"
  num_topics=100

  print "Removing existing files from directory."
  delete_models(fieldname)

  print "Dictionary Test"
  dictionary = None
  dictionary = load(build_dictionary_path(fieldname))
  print "Dictionary: Should be none"
  if dictionary:
    print "DICTIONARY SHOULD NOT HAVE VALUES!"
  else:
    print dictionary

  corpus = lsa_models.getCorpus(fieldname)
  dictionary = lsa_models.getDictionary(corpus, fieldname)
  save(build_dictionary_path(fieldname), dictionary)
  dictionary = None
  dictionary = load(build_dictionary_path(fieldname))
  print "Dictionary: Should have values"
  print dictionary.values()[:10]

  print "TFIDF Test"
  tifdf = None
  tfidf = load(build_tfidf_path(fieldname))
  print "Tfidf: Should be none"
  print tfidf
  
  tfidf = lsa_models.getTfidfModel(corpus, fieldname)
  save(build_tfidf_path(fieldname), tfidf)
  tfidf = None
  tfidf = load(build_tfidf_path(fieldname))
  print "tfidf: Should have values"
  print tfidf

  print "LSI Test"
  lsi = None
  lsi = load(build_lsi_path(fieldname))
  print "LSI: Should be none"
  print lsi
  
  lsi = lsa_models.getLsiModel(corpus, fieldname, num_topics)
  save(build_lsi_path(fieldname), lsi)
  lsi = None
  lsi = load(build_lsi_path(fieldname))
  print "lsi: Should have values"
  print lsi


if __name__=="__main__":
    main()
