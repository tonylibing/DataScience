#!/usr/bin/env python
# encoding: utf-8

import os
from os.path import join

HOME= '/'.join(os.popen('pwd').read().split())

BIN= os.path.join(HOME, 'bin')
DATA= os.path.join(HOME, 'data')

TRAIN_CSV = os.path.join(DATA, 'train.csv')
AD_CSV = join(DATA, 'ad.csv')
POSITION_CSV =join(DATA, 'position.csv')
TRAIN_CSV = join(DATA, 'train.csv')
USER_APP_ACTION = join(DATA, 'user_app_actions.csv')
APP_CATEGORIES_CSV =join(DATA,'app_categories.csv')
TEST_CSV = join(DATA, 'test.csv')
USER_CSV = join(DATA, 'user.csv')
USER_INSTALLEDAPPS_CSV = join(DATA, 'user_installedapps.csv')
print TEST_CSV
