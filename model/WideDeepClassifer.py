# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from collections import namedtuple
from itertools import chain
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

pd.options.mode.chained_assignment = None


def label_encode(df, cols=None):
    """
    Helper function to label-encode some features of a given dataset.

    Parameters:
    --------
    df  (pd.Dataframe)
    cols (list): optional - columns to be label-encoded

    Returns:
    ________
    val_to_idx (dict) : Dictionary of dictionaries with useful information about
    the encoding mapping
    df (pd.Dataframe): mutated df with Label-encoded features.
    """

    if cols == None:
        cols = list(df.select_dtypes(include=['object']).columns)

    val_types = dict()
    for c in cols:
        val_types[c] = df[c].unique()

    val_to_idx = dict()
    for k, v in val_types.iteritems():
        val_to_idx[k] = {o: i for i, o in enumerate(val_types[k])}

    for k, v in val_to_idx.iteritems():
        df[k] = df[k].apply(lambda x: v[x])

    return val_to_idx, df


def prepare_data(df, wide_cols, crossed_cols, embeddings_cols, continuous_cols, target,
                 scale=False, def_dim=8, seed=1981):
    """Prepares a pandas dataframe for the WideDeep model.

    Parameters:
    ----------
    df (pd.Dataframe)
    wide_cols : list with the columns to be used for the wide-side of the model
    crossed_cols : list of tuples with the columns to be crossed
    embeddings_cols : this can be a list of column names or a list of tuples with
    2 elements: (col_name, embedding dimension for this column)
    continuous_cols : list with the continous column names
    target (str) : the target to be fitted
    scale (bool) : boolean indicating if the continuous columns must be scaled
    def_dim (int) : Default dimension of the embeddings. If no embedding dimension is
    included in the "embeddings_cols" input all embedding columns will use this value (8)
    seed (int) : Random State for the train/test split

    Returns:
    ----------
    wd_dataset (dict): dict with:
    train_dataset/test_dataset: tuples with the wide, deep and lable training and
    testing datasets
    encoding_dict : dict with useful information about the encoding of the features.
    For example, given a feature 'education' and a value for that feature 'Doctorate'
    encoding_dict['education']['Doctorate'] will return an the encoded integer.
    embeddings_input : list of tuples with the embeddings info per column:
    ('col_name', number of unique values, embedding dimension)
    deep_column_idx : dict with the column indexes of all columns considerd in the Deep-Side
    of the model
    """

    # If embeddings_cols does not include the embeddings dimensions it will be set as
    # def_dim
    if type(embeddings_cols[0]) is tuple:
        emb_dim = dict(embeddings_cols)
        embeddings_cols = [emb[0] for emb in embeddings_cols]
    else:
        emb_dim = {e: def_dim for e in embeddings_cols}
    deep_cols = embeddings_cols + continuous_cols

    # Extract the target and copy the dataframe so we don't mutate it
    # internally.
    Y = np.array(df[target])
    all_columns = list(set(wide_cols + deep_cols + list(chain(*crossed_cols))))
    df_tmp = df.copy()[all_columns]

    # Build the crossed columns
    crossed_columns = []
    for cols in crossed_cols:
        colname = '_'.join(cols)
        df_tmp[colname] = df_tmp[cols].apply(lambda x: '-'.join(x), axis=1)
        crossed_columns.append(colname)

    #  Extract the categorical column names that can be one hot encoded later
    categorical_columns = list(df_tmp.select_dtypes(include=['object']).columns)

    # Encode the dataframe and get the encoding Dictionary only for the
    # deep_cols (for the wide_cols is uneccessary)
    encoding_dict, df_tmp = label_encode(df_tmp)
    encoding_dict = {k: encoding_dict[k] for k in encoding_dict if k in deep_cols}
    embeddings_input = []
    for k, v in encoding_dict.iteritems():
        embeddings_input.append((k, len(v), emb_dim[k]))

    # select the deep_cols and get the column index that will be use later
    # to slice the tensors
    df_deep = df_tmp[deep_cols]
    deep_column_idx = {k: v for v, k in enumerate(df_deep.columns)}

    # The continous columns will be concatenated with the embeddings, so you
    # probably want to normalize them first
    if scale:
        scaler = StandardScaler()
        for cc in continuous_cols:
            df_deep[cc] = scaler.fit_transform(df_deep[cc].values.reshape(-1, 1))

    # select the wide_cols and one-hot encode those that are categorical
    df_wide = df_tmp[wide_cols + crossed_columns]
    del (df_tmp)
    dummy_cols = [c for c in wide_cols + crossed_columns if c in categorical_columns]
    df_wide = pd.get_dummies(df_wide, columns=dummy_cols)

    # train/test split
    X_train_deep, X_test_deep = train_test_split(df_deep.values, test_size=0.3, random_state=seed)
    X_train_wide, X_test_wide = train_test_split(df_wide.values, test_size=0.3, random_state=seed)
    y_train, y_test = train_test_split(Y, test_size=0.3, random_state=1981)

    # Building the output dictionary
    wd_dataset = dict()
    train_dataset = namedtuple('train_dataset', 'wide, deep, labels')
    test_dataset = namedtuple('test_dataset', 'wide, deep, labels')
    wd_dataset['train_dataset'] = train_dataset(X_train_wide, X_train_deep, y_train)
    wd_dataset['test_dataset'] = test_dataset(X_test_wide, X_test_deep, y_test)
    wd_dataset['embeddings_input'] = embeddings_input
    wd_dataset['deep_column_idx'] = deep_column_idx
    wd_dataset['encoding_dict'] = encoding_dict

    return wd_dataset


use_cuda = torch.cuda.is_available()


class WideDeepLoader(Dataset):
    """Helper to facilitate loading the data to the pytorch models.

    Parameters:
    --------
    data: namedtuple with 3 elements - (wide_input_data, deep_inp_data, target)
    """

    def __init__(self, data):
        self.X_wide = data.wide
        self.X_deep = data.deep
        self.Y = data.labels

    def __getitem__(self, idx):
        xw = self.X_wide[idx]
        xd = self.X_deep[idx]
        y = self.Y[idx]

        return xw, xd, y

    def __len__(self):
        return len(self.Y)


class WideDeep(nn.Module):
    """ Wide and Deep model. As explained in Heng-Tze Cheng et al., 2016, the
    model taked the wide features and the deep features after being passed through
    the hidden layers and connects them to an output neuron. For details, please
    refer to the paper and the corresponding tutorial in the tensorflow site:
    https://www.tensorflow.org/tutorials/wide_and_deep

    Parameters:
    --------
    wide_dim (int) : dim of the wide-side input tensor
    embeddings_input (tuple): 3-elements tuple with the embeddings "set-up" -
    (col_name, unique_values, embeddings dim)
    continuous_cols (list) : list with the name of the continuum columns
    deep_column_idx (dict) : dictionary where the keys are column names and the values
    their corresponding index in the deep-side input tensor
    hidden_layers (list) : list with the number of units per hidden layer
    encoding_dict (dict) : dictionary with the label-encode mapping
    n_class (int) : number of classes. Defaults to 1 if logistic or regression
    dropout (float)
    """

    def __init__(self,
                 wide_dim,
                 embeddings_input,
                 continuous_cols,
                 deep_column_idx,
                 hidden_layers,
                 dropout,
                 encoding_dict,
                 n_class):

        super(WideDeep, self).__init__()
        self.wide_dim = wide_dim
        self.deep_column_idx = deep_column_idx
        self.embeddings_input = embeddings_input
        self.continuous_cols = continuous_cols
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.encoding_dict = encoding_dict
        self.n_class = n_class

        # Build the embedding layers to be passed through the deep-side
        for col, val, dim in self.embeddings_input:
            setattr(self, 'emb_layer_' + col, nn.Embedding(val, dim))

        # Build the deep-side hidden layers with dropout if specified
        input_emb_dim = np.sum([emb[2] for emb in self.embeddings_input])
        self.linear_1 = nn.Linear(input_emb_dim + len(continuous_cols), self.hidden_layers[0])
        if self.dropout:
            self.linear_1_drop = nn.Dropout(self.dropout[0])
        for i, h in enumerate(self.hidden_layers[1:], 1):
            setattr(self, 'linear_' + str(i + 1), nn.Linear(self.hidden_layers[i - 1], self.hidden_layers[i]))
            if self.dropout:
                setattr(self, 'linear_' + str(i + 1) + '_drop', nn.Dropout(self.dropout[i]))

        # Connect the wide- and dee-side of the model to the output neuron(s)
        self.output = nn.Linear(self.hidden_layers[-1] + self.wide_dim, self.n_class)

    def compile(self, method="logistic", optimizer="Adam", learning_rate=0.001, momentum=0.0):
        """Wrapper to set the activation, loss and the optimizer.

        Parameters:
        ----------
        method (str) : regression, logistic or multiclass
        optimizer (str): SGD, Adam, or RMSprop
        """
        if method == 'regression':
            self.activation, self.criterion = None, F.mse_loss
        if method == 'logistic':
            self.activation, self.criterion = F.sigmoid, F.binary_cross_entropy
        if method == 'multiclass':
            self.activation, self.criterion = F.softmax, F.cross_entropy

        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        if optimizer == "RMSprop":
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate)
        if optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)

        self.method = method

    def forward(self, X_w, X_d):
        """Implementation of the forward pass.

        Parameters:
        ----------
        X_w (torch.tensor) : wide-side input tensor
        X_d (torch.tensor) : deep-side input tensor

        Returns:
        --------
        out (torch.tensor) : result of the output neuron(s)
        """
        # Deep Side
        emb = [getattr(self, 'emb_layer_' + col)(X_d[:, self.deep_column_idx[col]].long())
               for col, _, _ in self.embeddings_input]
        if self.continuous_cols:
            cont_idx = [self.deep_column_idx[col] for col in self.continuous_cols]
            cont = [X_d[:, cont_idx].float()]
            deep_inp = torch.cat(emb + cont, 1)
        else:
            deep_inp = torch.cat(emb, 1)

        x_deep = F.relu(self.linear_1(deep_inp))
        if self.dropout:
            x_deep = self.linear_1_drop(x_deep)
        for i in range(1, len(self.hidden_layers)):
            x_deep = F.relu(getattr(self, 'linear_' + str(i + 1))(x_deep))
            if self.dropout:
                x_deep = getattr(self, 'linear_' + str(i + 1) + '_drop')(x_deep)

        # Deep + Wide sides
        wide_deep_input = torch.cat([x_deep, X_w.float()], 1)

        if not self.activation:
            out = self.output(wide_deep_input)
        else:
            out = self.activation(self.output(wide_deep_input))

        return out

    def fit(self, dataset, n_epochs, batch_size):
        """Run the model for the training set at dataset.

        Parameters:
        ----------
        dataset (dict): dictionary with the training sets -
        X_wide_train, X_deep_train, target
        n_epochs (int)
        batch_size (int)
        """
        widedeep_dataset = WideDeepLoader(dataset)
        train_loader = torch.utils.data.DataLoader(dataset=widedeep_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        # set the model in training mode
        net = self.train()
        for epoch in range(n_epochs):
            total = 0
            correct = 0
            for i, (X_wide, X_deep, target) in enumerate(train_loader):
                X_w = Variable(X_wide)
                X_d = Variable(X_deep)
                y = (Variable(target).float() if self.method != 'multiclass' else Variable(target))

                if use_cuda:
                    X_w, X_d, y = X_w.cuda(), X_d.cuda(), y.cuda()

                self.optimizer.zero_grad()
                y_pred = net(X_w, X_d)
                loss = self.criterion(y_pred, y)
                loss.backward()
                self.optimizer.step()

                if self.method != "regression":
                    total += y.size(0)
                    if self.method == 'logistic':
                        y_pred_cat = (y_pred > 0.5).squeeze(1).float()
                    if self.method == "multiclass":
                        _, y_pred_cat = torch.max(y_pred, 1)
                    correct += float((y_pred_cat == y).sum().data[0])

            if self.method != "regression":
                print ('Epoch {} of {}, Loss: {}, accuracy: {}'.format(epoch + 1,
                                                                       n_epochs, round(loss.data[0], 3),
                                                                       round(correct / total, 4)))
            else:
                print ('Epoch {} of {}, Loss: {}'.format(epoch + 1, n_epochs,
                                                         round(loss.data[0], 3)))

    def predict(self, dataset):
        """Predict target for dataset.

        Parameters:
        ----------
        dataset (dict): dictionary with the testing dataset -
        X_wide_test, X_deep_test, target

        Returns:
        --------
        array-like with the target for dataset
        """

        X_w = Variable(torch.from_numpy(dataset.wide)).float()
        X_d = Variable(torch.from_numpy(dataset.deep))

        if use_cuda:
            X_w, X_d = X_w.cuda(), X_d.cuda()

        # set the model in evaluation mode so dropout is not applied
        net = self.eval()
        pred = net(X_w, X_d).cpu()
        if self.method == "regression":
            return pred.squeeze(1).data.numpy()
        if self.method == "logistic":
            return (pred > 0.5).squeeze(1).data.numpy()
        if self.method == "multiclass":
            _, pred_cat = torch.max(pred, 1)
            return pred_cat.data.numpy()

    def predict_proba(self, dataset):
        """Predict predict probability for dataset.
        This method will only work with method logistic/multiclass

        Parameters:
        ----------
        dataset (dict): dictionary with the testing dataset -
        X_wide_test, X_deep_test, target

        Returns:
        --------
        array-like with the probability for dataset.
        """

        X_w = Variable(torch.from_numpy(dataset.wide)).float()
        X_d = Variable(torch.from_numpy(dataset.deep))

        if use_cuda:
            X_w, X_d = X_w.cuda(), X_d.cuda()

        # set the model in evaluation mode so dropout is not applied
        net = self.eval()
        pred = net(X_w, X_d).cpu()
        if self.method == "logistic":
            pred = pred.squeeze(1).data.numpy()
            probs = np.zeros([pred.shape[0], 2])
            probs[:, 0] = 1 - pred
            probs[:, 1] = pred
            return probs
        if self.method == "multiclass":
            return pred.data.numpy()

    def get_embeddings(self, col_name):
        """Extract the embeddings for the embedding columns.

        Parameters:
        -----------
        col_name (str) : column we want the embedding for

        Returns:
        --------
        embeddings_dict (dict): dictionary with the column values and the embeddings
        """

        params = list(self.named_parameters())
        emb_layers = [p for p in params if 'emb_layer' in p[0]]
        emb_layer = [layer for layer in emb_layers if col_name in layer[0]][0]
        embeddings = emb_layer[1].cpu().data.numpy()
        col_label_encoding = self.encoding_dict[col_name]
        inv_dict = {v: k for k, v in col_label_encoding.iteritems()}
        embeddings_dict = {}
        for idx, value in inv_dict.iteritems():
            embeddings_dict[value] = embeddings[idx]

        return embeddings_dict
