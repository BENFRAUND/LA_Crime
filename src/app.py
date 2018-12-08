import pandas as pd
from datetime import date
import requests
from lacrime_config import lacrime_api_key
import sqlite3
import sys
from flask import Flask, render_template,jsonify,request, redirect
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
import numpy as np
import keras
from keras import backend as K
from keras.models import load_model

app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///db/la_crime.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

Base = automap_base(metadata=db.metadata)
engine = db.get_engine()
Base.prepare(engine, reflect=True)
LACrime = Base.classes.new_la_crime_incl_modesc
ZipCombinedStats = Base.classes.la_zip_stats
CrimeAccuracy = Base.classes.crime_accuracy_cnt

crime_model = None
graph = None

def load_model():
    global crime_model
    global graph
    crime_model = keras.models.load_model("ml_model/DeepNN_model/fbi_category_model_trained.h5")
    graph = K.get_session().graph

load_model()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/new_la_crime_data")
def refresh_data():
    # Collect refreshed data from LA crime API (https://data.lacity.org/resource/7fvc-faax.json?)
    incidents = requests.get("https://data.lacity.org/resource/7fvc-faax.json?$$app_token=" + lacrime_api_key + "&$order=date_occ%20DESC&$limit=2000").json()
    
    # Store new crime_data in dataframe and convert datatypes
    crime_data = pd.DataFrame(incidents).set_index(["dr_no"])
    crime_data = crime_data.sort_values(by='date_occ', ascending=False)

    crime_data['area_id'] = crime_data['area_id'].apply(pd.to_numeric)
    crime_data['crm_cd'] = crime_data['crm_cd'].apply(pd.to_numeric)
    crime_data = crime_data.drop(columns=['crm_cd_1'])
    crime_data = crime_data.drop(columns=['crm_cd_2'])
    #crime_data = crime_data.drop(columns=['crm_cd_3'])
    #crime_data = crime_data.drop(columns=['crm_cd_4'])
    crime_data['date_occ'] = pd.to_datetime(crime_data['date_occ'], infer_datetime_format=True)
    crime_data['date_rptd'] = pd.to_datetime(crime_data['date_rptd'], infer_datetime_format=True)
    crime_data[['loc_type','coordinates']] = crime_data['location_1'].apply(pd.Series)
    crime_data[['longitude','latitude']] = crime_data['coordinates'].apply(pd.Series)
    crime_data = crime_data.drop(columns=['location_1'])
    crime_data = crime_data.drop(columns=['coordinates'])
    crime_data['premis_cd'] = crime_data['premis_cd'].apply(pd.to_numeric)
    crime_data['rpt_dist_no'] = crime_data['rpt_dist_no'].apply(pd.to_numeric)
    crime_data['hour_occ'] = crime_data['time_occ'].str[:2].apply(pd.to_numeric)
    crime_data['minute_occ'] = crime_data['time_occ'].str[2:].apply(pd.to_numeric)
    crime_data = crime_data.drop(columns=['time_occ'])
    crime_data['vict_age'] = crime_data['vict_age'].apply(pd.to_numeric)
    crime_data['weapon_used_cd'] = crime_data['weapon_used_cd'].apply(pd.to_numeric)
    

    # Drop and create new_la_crime table in sqlite db form crime_data dataframe
    # Create sqlalchemy engine
    from sqlalchemy import create_engine
    conn = sqlite3.connect("db/la_crime.db")
    c = conn.cursor()
    c.execute('''DROP TABLE IF EXISTS new_la_crime;''')
    c.execute('''CREATE TABLE new_la_crime
                (dr_no INTEGER PRIMARY KEY,
                area_id INTEGER,
                area_name VARCHAR(64),
                crm_cd INTEGER,
                crm_cd_1 INTEGER,
                crm_cd_2 INTEGER,
                crm_cd_3 INTEGER,
                crm_cd_desc VARCHAR(64),
                cross_street VARCHAR(64),
                date_occ DATE,
                date_rptd DATE,
                location VARCHAR(64),
                longitude FLOAT,
                latitude FLOAT,
                mocodes TEXT,
                premis_cd INTEGER,
                premis_desc VARCHAR(64),
                rpt_dist_no INTEGER,
                status VARCHAR(2),
                status_desc VARCHAR(64),
                hour_occ INTEGER,
                minute_occ INTEGER,
                vict_age INTEGER,
                vict_descent VARCHAR(64),
                vict_sex VARCHAR(2),
                weapon_desc VARCHAR(64),
                weapon_used_cd INTEGER,
                loc_type VARCHAR(64))
                ''')
    conn.commit()
    conn.close()

    # Create new_crime_table
    engine = create_engine('sqlite:///db/la_crime.db')
    crime_data.to_sql('new_la_crime', engine, if_exists='append', index=True)
    
    # Create new crime table including FBI categories
    from sqlalchemy import create_engine

    conn = sqlite3.connect("db/la_crime.db")
    c = conn.cursor()
    c.execute('''drop table if exists la_crime_incl_fbi_cat;''')
    c.execute('''
                create table la_crime_incl_fbi_cat as select
                dr_no,
                area_id,
                area_name,
                la.crm_cd,
                la.crm_cd_desc,
                cross_street,
                cast(strftime('%Y%m%d', date_occ) as integer) as date_occ,
                cast(strftime('%Y%m%d', date_rptd) as integer) as date_rptd,
                location,
                longitude,
                latitude,
                ('["'||replace(mocodes,' ','", "')||'"]') as mocodes,
                premis_cd,
                premis_desc,
                rpt_dist_no,
                status,
                status_desc,
                hour_occ,
                minute_occ,
                vict_age,
                vict_descent,
                vict_sex,
                weapon_desc,
                weapon_used_cd,
                loc_type,
                FBI_Category 
                from new_la_crime as la
                left join fbi_cat_tmp2 as fb
                on la.crm_cd = fb.crm_cd
                where fbi_part_1 !="Other"
                group by 
                dr_no,
                area_id,
                area_name,
                la.crm_cd,
                la.crm_cd_desc,
                cross_street,
                cast(strftime('%Y%m%d', date_occ) as integer),
                cast(strftime('%Y%m%d', date_rptd) as integer),
                location,
                longitude,
                latitude,
                ('["'||replace(mocodes,' ','", "')||'"]'),
                premis_cd,
                premis_desc,
                rpt_dist_no,
                status,
                status_desc,
                hour_occ,
                minute_occ,
                vict_age,
                vict_descent,
                vict_sex,
                weapon_desc,
                weapon_used_cd,
                loc_type,
                FBI_Category
                  ''')
    conn.commit()
    conn.close()

    conn = sqlite3.connect("db/la_crime.db")
    c = conn.cursor()
    c.execute('''drop table if exists new_ml_8_cat_crime_data''')
    c.execute('''create table new_ml_8_cat_crime_data as select
                cast(dr_no as INT) as dr_no,
                cast(area_id as INT) as area_id,
                cast(date_occ as INT) as date_occ,
                cast(date_rptd as INT) as date_rptd,
                cast(longitude as FLOAT64) as longitude,
                cast(latitude as FLOAT64) as latitude,
                cast(premis_cd as FLOAT64) as premis_cd,
                cast(rpt_dist_no as INT) as rpt_dist_no,
                cast(hour_occ as INT) as hour_occ,
                cast(minute_occ as INT) as minute_occ,
                cast(vict_age as FLOAT64) as vict_age,
                FBI_Category,
                cast((case when vict_descent is null then 1 else 0 end) as INT) as '-(vict_descent)',
                cast((case when vict_descent = 'A' then 1 else 0 end) as INT) as 'A(vict_descent)',
                cast((case when vict_descent = 'B' then 1 else 0 end) as INT) as 'B(vict_descent)',
                cast((case when vict_descent = 'C' then 1 else 0 end) as INT) as 'C(vict_descent)',
                cast((case when vict_descent = 'D' then 1 else 0 end) as INT) as 'D(vict_descent)',
                cast((case when vict_descent = 'F' then 1 else 0 end) as INT) as 'F(vict_descent)',
                cast((case when vict_descent = 'G' then 1 else 0 end) as INT) as 'G(vict_descent)',
                cast((case when vict_descent = 'H' then 1 else 0 end) as INT) as 'H(vict_descent)',
                cast((case when vict_descent = 'I' then 1 else 0 end) as INT) as 'I(vict_descent)',
                cast((case when vict_descent = 'J' then 1 else 0 end) as INT) as 'J(vict_descent)',
                cast((case when vict_descent = 'K' then 1 else 0 end) as INT) as 'K(vict_descent)',
                cast((case when vict_descent = 'L' then 1 else 0 end) as INT) as 'L(vict_descent)',
                cast((case when vict_descent = 'O' then 1 else 0 end) as INT) as 'O(vict_descent)',
                cast((case when vict_descent = 'P' then 1 else 0 end) as INT) as 'P(vict_descent)',
                cast((case when vict_descent = 'S' then 1 else 0 end) as INT) as 'S(vict_descent)',
                cast((case when vict_descent = 'U' then 1 else 0 end) as INT) as 'U(vict_descent)',
                cast((case when vict_descent = 'V' then 1 else 0 end) as INT) as 'V(vict_descent)',
                cast((case when vict_descent = 'W' then 1 else 0 end) as INT) as 'W(vict_descent)',
                cast((case when vict_descent = 'X' then 1 else 0 end) as INT) as 'X(vict_descent)',
                cast((case when vict_descent = 'Z' then 1 else 0 end) as INT) as 'Z(vict_descent)',
                cast((case when vict_sex is null then 1 else 0 end) as INT) as '-(vict_sex)',
                cast((case when vict_sex = 'F' then 1 else 0 end) as INT) as 'F(vict_sex)',
                cast((case when vict_sex = 'H' then 1 else 0 end) as INT) as 'H(vict_sex)',
                cast((case when vict_sex = 'M' then 1 else 0 end) as INT) as 'M(vict_sex)',
                cast((case when vict_sex = 'N' then 1 else 0 end) as INT) as 'N(vict_sex)',
                cast((case when vict_sex = 'X' then 1 else 0 end) as INT) as 'X(vict_sex)',
                cast((case when status is null then 1 else 0 end) as INT) as '-(status)',
                cast((case when status = '13' then 1 else 0 end) as INT) as '13(status)',
                cast((case when status = '19' then 1 else 0 end) as INT) as '19(status)',
                cast((case when status = 'AA' then 1 else 0 end) as INT) as 'AA(status)',
                cast((case when status = 'AO' then 1 else 0 end) as INT) as 'AO(status)',
                cast((case when status = 'CC' then 1 else 0 end) as INT) as 'CC(status)',
                cast((case when status = 'IC' then 1 else 0 end) as INT) as 'IC(status)',
                cast((case when status = 'JA' then 1 else 0 end) as INT) as 'JA(status)',
                cast((case when status = 'JO' then 1 else 0 end) as INT) as 'JO(status)',
                cast((case when status = 'TH' then 1 else 0 end) as INT) as 'TH(status)',
                cast((case when mocodes like '%0100%' then 1 else 0 end) as INT) as '0100',
                cast((case when mocodes like '%0101%' then 1 else 0 end) as INT) as '0101',
                cast((case when mocodes like '%0102%' then 1 else 0 end) as INT) as '0102',
                cast((case when mocodes like '%0103%' then 1 else 0 end) as INT) as '0103',
                cast((case when mocodes like '%0104%' then 1 else 0 end) as INT) as '0104',
                cast((case when mocodes like '%0105%' then 1 else 0 end) as INT) as '0105',
                cast((case when mocodes like '%0106%' then 1 else 0 end) as INT) as '0106',
                cast((case when mocodes like '%0107%' then 1 else 0 end) as INT) as '0107',
                cast((case when mocodes like '%0108%' then 1 else 0 end) as INT) as '0108',
                cast((case when mocodes like '%0109%' then 1 else 0 end) as INT) as '0109',
                cast((case when mocodes like '%0110%' then 1 else 0 end) as INT) as '0110',
                cast((case when mocodes like '%0112%' then 1 else 0 end) as INT) as '0112',
                cast((case when mocodes like '%0113%' then 1 else 0 end) as INT) as '0113',
                cast((case when mocodes like '%0114%' then 1 else 0 end) as INT) as '0114',
                cast((case when mocodes like '%0115%' then 1 else 0 end) as INT) as '0115',
                cast((case when mocodes like '%0116%' then 1 else 0 end) as INT) as '0116',
                cast((case when mocodes like '%0117%' then 1 else 0 end) as INT) as '0117',
                cast((case when mocodes like '%0118%' then 1 else 0 end) as INT) as '0118',
                cast((case when mocodes like '%0119%' then 1 else 0 end) as INT) as '0119',
                cast((case when mocodes like '%0120%' then 1 else 0 end) as INT) as '0120',
                cast((case when mocodes like '%0121%' then 1 else 0 end) as INT) as '0121',
                cast((case when mocodes like '%0122%' then 1 else 0 end) as INT) as '0122',
                cast((case when mocodes like '%0123%' then 1 else 0 end) as INT) as '0123',
                cast((case when mocodes like '%0200%' then 1 else 0 end) as INT) as '0200',
                cast((case when mocodes like '%0201%' then 1 else 0 end) as INT) as '0201',
                cast((case when mocodes like '%0202%' then 1 else 0 end) as INT) as '0202',
                cast((case when mocodes like '%0203%' then 1 else 0 end) as INT) as '0203',
                cast((case when mocodes like '%0204%' then 1 else 0 end) as INT) as '0204',
                cast((case when mocodes like '%0205%' then 1 else 0 end) as INT) as '0205',
                cast((case when mocodes like '%0206%' then 1 else 0 end) as INT) as '0206',
                cast((case when mocodes like '%0207%' then 1 else 0 end) as INT) as '0207',
                cast((case when mocodes like '%0208%' then 1 else 0 end) as INT) as '0208',
                cast((case when mocodes like '%0209%' then 1 else 0 end) as INT) as '0209',
                cast((case when mocodes like '%0210%' then 1 else 0 end) as INT) as '0210',
                cast((case when mocodes like '%0211%' then 1 else 0 end) as INT) as '0211',
                cast((case when mocodes like '%0212%' then 1 else 0 end) as INT) as '0212',
                cast((case when mocodes like '%0213%' then 1 else 0 end) as INT) as '0213',
                cast((case when mocodes like '%0214%' then 1 else 0 end) as INT) as '0214',
                cast((case when mocodes like '%0215%' then 1 else 0 end) as INT) as '0215',
                cast((case when mocodes like '%0216%' then 1 else 0 end) as INT) as '0216',
                cast((case when mocodes like '%0217%' then 1 else 0 end) as INT) as '0217',
                cast((case when mocodes like '%0218%' then 1 else 0 end) as INT) as '0218',
                cast((case when mocodes like '%0219%' then 1 else 0 end) as INT) as '0219',
                cast((case when mocodes like '%0220%' then 1 else 0 end) as INT) as '0220',
                cast((case when mocodes like '%0301%' then 1 else 0 end) as INT) as '0301',
                cast((case when mocodes like '%0302%' then 1 else 0 end) as INT) as '0302',
                cast((case when mocodes like '%0303%' then 1 else 0 end) as INT) as '0303',
                cast((case when mocodes like '%0304%' then 1 else 0 end) as INT) as '0304',
                cast((case when mocodes like '%0305%' then 1 else 0 end) as INT) as '0305',
                cast((case when mocodes like '%0306%' then 1 else 0 end) as INT) as '0306',
                cast((case when mocodes like '%0307%' then 1 else 0 end) as INT) as '0307',
                cast((case when mocodes like '%0308%' then 1 else 0 end) as INT) as '0308',
                cast((case when mocodes like '%0309%' then 1 else 0 end) as INT) as '0309',
                cast((case when mocodes like '%0310%' then 1 else 0 end) as INT) as '0310',
                cast((case when mocodes like '%0311%' then 1 else 0 end) as INT) as '0311',
                cast((case when mocodes like '%0312%' then 1 else 0 end) as INT) as '0312',
                cast((case when mocodes like '%0313%' then 1 else 0 end) as INT) as '0313',
                cast((case when mocodes like '%0314%' then 1 else 0 end) as INT) as '0314',
                cast((case when mocodes like '%0315%' then 1 else 0 end) as INT) as '0315',
                cast((case when mocodes like '%0316%' then 1 else 0 end) as INT) as '0316',
                cast((case when mocodes like '%0317%' then 1 else 0 end) as INT) as '0317',
                cast((case when mocodes like '%0318%' then 1 else 0 end) as INT) as '0318',
                cast((case when mocodes like '%0319%' then 1 else 0 end) as INT) as '0319',
                cast((case when mocodes like '%0320%' then 1 else 0 end) as INT) as '0320',
                cast((case when mocodes like '%0321%' then 1 else 0 end) as INT) as '0321',
                cast((case when mocodes like '%0322%' then 1 else 0 end) as INT) as '0322',
                cast((case when mocodes like '%0323%' then 1 else 0 end) as INT) as '0323',
                cast((case when mocodes like '%0324%' then 1 else 0 end) as INT) as '0324',
                cast((case when mocodes like '%0325%' then 1 else 0 end) as INT) as '0325',
                cast((case when mocodes like '%0326%' then 1 else 0 end) as INT) as '0326',
                cast((case when mocodes like '%0327%' then 1 else 0 end) as INT) as '0327',
                cast((case when mocodes like '%0328%' then 1 else 0 end) as INT) as '0328',
                cast((case when mocodes like '%0329%' then 1 else 0 end) as INT) as '0329',
                cast((case when mocodes like '%0330%' then 1 else 0 end) as INT) as '0330',
                cast((case when mocodes like '%0331%' then 1 else 0 end) as INT) as '0331',
                cast((case when mocodes like '%0332%' then 1 else 0 end) as INT) as '0332',
                cast((case when mocodes like '%0333%' then 1 else 0 end) as INT) as '0333',
                cast((case when mocodes like '%0334%' then 1 else 0 end) as INT) as '0334',
                cast((case when mocodes like '%0335%' then 1 else 0 end) as INT) as '0335',
                cast((case when mocodes like '%0336%' then 1 else 0 end) as INT) as '0336',
                cast((case when mocodes like '%0337%' then 1 else 0 end) as INT) as '0337',
                cast((case when mocodes like '%0338%' then 1 else 0 end) as INT) as '0338',
                cast((case when mocodes like '%0339%' then 1 else 0 end) as INT) as '0339',
                cast((case when mocodes like '%0340%' then 1 else 0 end) as INT) as '0340',
                cast((case when mocodes like '%0341%' then 1 else 0 end) as INT) as '0341',
                cast((case when mocodes like '%0342%' then 1 else 0 end) as INT) as '0342',
                cast((case when mocodes like '%0343%' then 1 else 0 end) as INT) as '0343',
                cast((case when mocodes like '%0344%' then 1 else 0 end) as INT) as '0344',
                cast((case when mocodes like '%0345%' then 1 else 0 end) as INT) as '0345',
                cast((case when mocodes like '%0346%' then 1 else 0 end) as INT) as '0346',
                cast((case when mocodes like '%0347%' then 1 else 0 end) as INT) as '0347',
                cast((case when mocodes like '%0348%' then 1 else 0 end) as INT) as '0348',
                cast((case when mocodes like '%0349%' then 1 else 0 end) as INT) as '0349',
                cast((case when mocodes like '%0350%' then 1 else 0 end) as INT) as '0350',
                cast((case when mocodes like '%0351%' then 1 else 0 end) as INT) as '0351',
                cast((case when mocodes like '%0352%' then 1 else 0 end) as INT) as '0352',
                cast((case when mocodes like '%0353%' then 1 else 0 end) as INT) as '0353',
                cast((case when mocodes like '%0354%' then 1 else 0 end) as INT) as '0354',
                cast((case when mocodes like '%0355%' then 1 else 0 end) as INT) as '0355',
                cast((case when mocodes like '%0356%' then 1 else 0 end) as INT) as '0356',
                cast((case when mocodes like '%0357%' then 1 else 0 end) as INT) as '0357',
                cast((case when mocodes like '%0358%' then 1 else 0 end) as INT) as '0358',
                cast((case when mocodes like '%0359%' then 1 else 0 end) as INT) as '0359',
                cast((case when mocodes like '%0360%' then 1 else 0 end) as INT) as '0360',
                cast((case when mocodes like '%0361%' then 1 else 0 end) as INT) as '0361',
                cast((case when mocodes like '%0362%' then 1 else 0 end) as INT) as '0362',
                cast((case when mocodes like '%0363%' then 1 else 0 end) as INT) as '0363',
                cast((case when mocodes like '%0364%' then 1 else 0 end) as INT) as '0364',
                cast((case when mocodes like '%0365%' then 1 else 0 end) as INT) as '0365',
                cast((case when mocodes like '%0366%' then 1 else 0 end) as INT) as '0366',
                cast((case when mocodes like '%0367%' then 1 else 0 end) as INT) as '0367',
                cast((case when mocodes like '%0368%' then 1 else 0 end) as INT) as '0368',
                cast((case when mocodes like '%0369%' then 1 else 0 end) as INT) as '0369',
                cast((case when mocodes like '%0370%' then 1 else 0 end) as INT) as '0370',
                cast((case when mocodes like '%0371%' then 1 else 0 end) as INT) as '0371',
                cast((case when mocodes like '%0372%' then 1 else 0 end) as INT) as '0372',
                cast((case when mocodes like '%0373%' then 1 else 0 end) as INT) as '0373',
                cast((case when mocodes like '%0374%' then 1 else 0 end) as INT) as '0374',
                cast((case when mocodes like '%0375%' then 1 else 0 end) as INT) as '0375',
                cast((case when mocodes like '%0376%' then 1 else 0 end) as INT) as '0376',
                cast((case when mocodes like '%0377%' then 1 else 0 end) as INT) as '0377',
                cast((case when mocodes like '%0378%' then 1 else 0 end) as INT) as '0378',
                cast((case when mocodes like '%0379%' then 1 else 0 end) as INT) as '0379',
                cast((case when mocodes like '%0380%' then 1 else 0 end) as INT) as '0380',
                cast((case when mocodes like '%0381%' then 1 else 0 end) as INT) as '0381',
                cast((case when mocodes like '%0382%' then 1 else 0 end) as INT) as '0382',
                cast((case when mocodes like '%0383%' then 1 else 0 end) as INT) as '0383',
                cast((case when mocodes like '%0384%' then 1 else 0 end) as INT) as '0384',
                cast((case when mocodes like '%0385%' then 1 else 0 end) as INT) as '0385',
                cast((case when mocodes like '%0386%' then 1 else 0 end) as INT) as '0386',
                cast((case when mocodes like '%0387%' then 1 else 0 end) as INT) as '0387',
                cast((case when mocodes like '%0388%' then 1 else 0 end) as INT) as '0388',
                cast((case when mocodes like '%0389%' then 1 else 0 end) as INT) as '0389',
                cast((case when mocodes like '%0390%' then 1 else 0 end) as INT) as '0390',
                cast((case when mocodes like '%0391%' then 1 else 0 end) as INT) as '0391',
                cast((case when mocodes like '%0392%' then 1 else 0 end) as INT) as '0392',
                cast((case when mocodes like '%0393%' then 1 else 0 end) as INT) as '0393',
                cast((case when mocodes like '%0394%' then 1 else 0 end) as INT) as '0394',
                cast((case when mocodes like '%0395%' then 1 else 0 end) as INT) as '0395',
                cast((case when mocodes like '%0396%' then 1 else 0 end) as INT) as '0396',
                cast((case when mocodes like '%0397%' then 1 else 0 end) as INT) as '0397',
                cast((case when mocodes like '%0398%' then 1 else 0 end) as INT) as '0398',
                cast((case when mocodes like '%0399%' then 1 else 0 end) as INT) as '0399',
                cast((case when mocodes like '%0400%' then 1 else 0 end) as INT) as '0400',
                cast((case when mocodes like '%0401%' then 1 else 0 end) as INT) as '0401',
                cast((case when mocodes like '%0402%' then 1 else 0 end) as INT) as '0402',
                cast((case when mocodes like '%0403%' then 1 else 0 end) as INT) as '0403',
                cast((case when mocodes like '%0404%' then 1 else 0 end) as INT) as '0404',
                cast((case when mocodes like '%0405%' then 1 else 0 end) as INT) as '0405',
                cast((case when mocodes like '%0406%' then 1 else 0 end) as INT) as '0406',
                cast((case when mocodes like '%0407%' then 1 else 0 end) as INT) as '0407',
                cast((case when mocodes like '%0408%' then 1 else 0 end) as INT) as '0408',
                cast((case when mocodes like '%0409%' then 1 else 0 end) as INT) as '0409',
                cast((case when mocodes like '%0410%' then 1 else 0 end) as INT) as '0410',
                cast((case when mocodes like '%0411%' then 1 else 0 end) as INT) as '0411',
                cast((case when mocodes like '%0412%' then 1 else 0 end) as INT) as '0412',
                cast((case when mocodes like '%0413%' then 1 else 0 end) as INT) as '0413',
                cast((case when mocodes like '%0414%' then 1 else 0 end) as INT) as '0414',
                cast((case when mocodes like '%0415%' then 1 else 0 end) as INT) as '0415',
                cast((case when mocodes like '%0416%' then 1 else 0 end) as INT) as '0416',
                cast((case when mocodes like '%0417%' then 1 else 0 end) as INT) as '0417',
                cast((case when mocodes like '%0418%' then 1 else 0 end) as INT) as '0418',
                cast((case when mocodes like '%0419%' then 1 else 0 end) as INT) as '0419',
                cast((case when mocodes like '%0420%' then 1 else 0 end) as INT) as '0420',
                cast((case when mocodes like '%0421%' then 1 else 0 end) as INT) as '0421',
                cast((case when mocodes like '%0422%' then 1 else 0 end) as INT) as '0422',
                cast((case when mocodes like '%0423%' then 1 else 0 end) as INT) as '0423',
                cast((case when mocodes like '%0424%' then 1 else 0 end) as INT) as '0424',
                cast((case when mocodes like '%0425%' then 1 else 0 end) as INT) as '0425',
                cast((case when mocodes like '%0426%' then 1 else 0 end) as INT) as '0426',
                cast((case when mocodes like '%0427%' then 1 else 0 end) as INT) as '0427',
                cast((case when mocodes like '%0428%' then 1 else 0 end) as INT) as '0428',
                cast((case when mocodes like '%0429%' then 1 else 0 end) as INT) as '0429',
                cast((case when mocodes like '%0430%' then 1 else 0 end) as INT) as '0430',
                cast((case when mocodes like '%0431%' then 1 else 0 end) as INT) as '0431',
                cast((case when mocodes like '%0432%' then 1 else 0 end) as INT) as '0432',
                cast((case when mocodes like '%0433%' then 1 else 0 end) as INT) as '0433',
                cast((case when mocodes like '%0434%' then 1 else 0 end) as INT) as '0434',
                cast((case when mocodes like '%0435%' then 1 else 0 end) as INT) as '0435',
                cast((case when mocodes like '%0436%' then 1 else 0 end) as INT) as '0436',
                cast((case when mocodes like '%0437%' then 1 else 0 end) as INT) as '0437',
                cast((case when mocodes like '%0438%' then 1 else 0 end) as INT) as '0438',
                cast((case when mocodes like '%0439%' then 1 else 0 end) as INT) as '0439',
                cast((case when mocodes like '%0440%' then 1 else 0 end) as INT) as '0440',
                cast((case when mocodes like '%0441%' then 1 else 0 end) as INT) as '0441',
                cast((case when mocodes like '%0442%' then 1 else 0 end) as INT) as '0442',
                cast((case when mocodes like '%0443%' then 1 else 0 end) as INT) as '0443',
                cast((case when mocodes like '%0444%' then 1 else 0 end) as INT) as '0444',
                cast((case when mocodes like '%0445%' then 1 else 0 end) as INT) as '0445',
                cast((case when mocodes like '%0446%' then 1 else 0 end) as INT) as '0446',
                cast((case when mocodes like '%0447%' then 1 else 0 end) as INT) as '0447',
                cast((case when mocodes like '%0448%' then 1 else 0 end) as INT) as '0448',
                cast((case when mocodes like '%0449%' then 1 else 0 end) as INT) as '0449',
                cast((case when mocodes like '%0450%' then 1 else 0 end) as INT) as '0450',
                cast((case when mocodes like '%0500%' then 1 else 0 end) as INT) as '0500',
                cast((case when mocodes like '%0501%' then 1 else 0 end) as INT) as '0501',
                cast((case when mocodes like '%0502%' then 1 else 0 end) as INT) as '0502',
                cast((case when mocodes like '%0503%' then 1 else 0 end) as INT) as '0503',
                cast((case when mocodes like '%0504%' then 1 else 0 end) as INT) as '0504',
                cast((case when mocodes like '%0505%' then 1 else 0 end) as INT) as '0505',
                cast((case when mocodes like '%0506%' then 1 else 0 end) as INT) as '0506',
                cast((case when mocodes like '%0507%' then 1 else 0 end) as INT) as '0507',
                cast((case when mocodes like '%0508%' then 1 else 0 end) as INT) as '0508',
                cast((case when mocodes like '%0509%' then 1 else 0 end) as INT) as '0509',
                cast((case when mocodes like '%0510%' then 1 else 0 end) as INT) as '0510',
                cast((case when mocodes like '%0511%' then 1 else 0 end) as INT) as '0511',
                cast((case when mocodes like '%0512%' then 1 else 0 end) as INT) as '0512',
                cast((case when mocodes like '%0513%' then 1 else 0 end) as INT) as '0513',
                cast((case when mocodes like '%0514%' then 1 else 0 end) as INT) as '0514',
                cast((case when mocodes like '%0515%' then 1 else 0 end) as INT) as '0515',
                cast((case when mocodes like '%0516%' then 1 else 0 end) as INT) as '0516',
                cast((case when mocodes like '%0517%' then 1 else 0 end) as INT) as '0517',
                cast((case when mocodes like '%0518%' then 1 else 0 end) as INT) as '0518',
                cast((case when mocodes like '%0519%' then 1 else 0 end) as INT) as '0519',
                cast((case when mocodes like '%0520%' then 1 else 0 end) as INT) as '0520',
                cast((case when mocodes like '%0521%' then 1 else 0 end) as INT) as '0521',
                cast((case when mocodes like '%0522%' then 1 else 0 end) as INT) as '0522',
                cast((case when mocodes like '%0523%' then 1 else 0 end) as INT) as '0523',
                cast((case when mocodes like '%0524%' then 1 else 0 end) as INT) as '0524',
                cast((case when mocodes like '%0525%' then 1 else 0 end) as INT) as '0525',
                cast((case when mocodes like '%0526%' then 1 else 0 end) as INT) as '0526',
                cast((case when mocodes like '%0527%' then 1 else 0 end) as INT) as '0527',
                cast((case when mocodes like '%0528%' then 1 else 0 end) as INT) as '0528',
                cast((case when mocodes like '%0529%' then 1 else 0 end) as INT) as '0529',
                cast((case when mocodes like '%0530%' then 1 else 0 end) as INT) as '0530',
                cast((case when mocodes like '%0531%' then 1 else 0 end) as INT) as '0531',
                cast((case when mocodes like '%0532%' then 1 else 0 end) as INT) as '0532',
                cast((case when mocodes like '%0533%' then 1 else 0 end) as INT) as '0533',
                cast((case when mocodes like '%0534%' then 1 else 0 end) as INT) as '0534',
                cast((case when mocodes like '%0535%' then 1 else 0 end) as INT) as '0535',
                cast((case when mocodes like '%0536%' then 1 else 0 end) as INT) as '0536',
                cast((case when mocodes like '%0537%' then 1 else 0 end) as INT) as '0537',
                cast((case when mocodes like '%0538%' then 1 else 0 end) as INT) as '0538',
                cast((case when mocodes like '%0539%' then 1 else 0 end) as INT) as '0539',
                cast((case when mocodes like '%0540%' then 1 else 0 end) as INT) as '0540',
                cast((case when mocodes like '%0541%' then 1 else 0 end) as INT) as '0541',
                cast((case when mocodes like '%0542%' then 1 else 0 end) as INT) as '0542',
                cast((case when mocodes like '%0543%' then 1 else 0 end) as INT) as '0543',
                cast((case when mocodes like '%0544%' then 1 else 0 end) as INT) as '0544',
                cast((case when mocodes like '%0545%' then 1 else 0 end) as INT) as '0545',
                cast((case when mocodes like '%0546%' then 1 else 0 end) as INT) as '0546',
                cast((case when mocodes like '%0547%' then 1 else 0 end) as INT) as '0547',
                cast((case when mocodes like '%0548%' then 1 else 0 end) as INT) as '0548',
                cast((case when mocodes like '%0549%' then 1 else 0 end) as INT) as '0549',
                cast((case when mocodes like '%0550%' then 1 else 0 end) as INT) as '0550',
                cast((case when mocodes like '%0551%' then 1 else 0 end) as INT) as '0551',
                cast((case when mocodes like '%0552%' then 1 else 0 end) as INT) as '0552',
                cast((case when mocodes like '%0553%' then 1 else 0 end) as INT) as '0553',
                cast((case when mocodes like '%0554%' then 1 else 0 end) as INT) as '0554',
                cast((case when mocodes like '%0555%' then 1 else 0 end) as INT) as '0555',
                cast((case when mocodes like '%0556%' then 1 else 0 end) as INT) as '0556',
                cast((case when mocodes like '%0557%' then 1 else 0 end) as INT) as '0557',
                cast((case when mocodes like '%0558%' then 1 else 0 end) as INT) as '0558',
                cast((case when mocodes like '%0559%' then 1 else 0 end) as INT) as '0559',
                cast((case when mocodes like '%0560%' then 1 else 0 end) as INT) as '0560',
                cast((case when mocodes like '%0561%' then 1 else 0 end) as INT) as '0561',
                cast((case when mocodes like '%0562%' then 1 else 0 end) as INT) as '0562',
                cast((case when mocodes like '%0563%' then 1 else 0 end) as INT) as '0563',
                cast((case when mocodes like '%0601%' then 1 else 0 end) as INT) as '0601',
                cast((case when mocodes like '%0602%' then 1 else 0 end) as INT) as '0602',
                cast((case when mocodes like '%0603%' then 1 else 0 end) as INT) as '0603',
                cast((case when mocodes like '%0604%' then 1 else 0 end) as INT) as '0604',
                cast((case when mocodes like '%0605%' then 1 else 0 end) as INT) as '0605',
                cast((case when mocodes like '%0701%' then 1 else 0 end) as INT) as '0701',
                cast((case when mocodes like '%0800%' then 1 else 0 end) as INT) as '0800',
                cast((case when mocodes like '%0901%' then 1 else 0 end) as INT) as '0901',
                cast((case when mocodes like '%0902%' then 1 else 0 end) as INT) as '0902',
                cast((case when mocodes like '%0903%' then 1 else 0 end) as INT) as '0903',
                cast((case when mocodes like '%0904%' then 1 else 0 end) as INT) as '0904',
                cast((case when mocodes like '%0905%' then 1 else 0 end) as INT) as '0905',
                cast((case when mocodes like '%0906%' then 1 else 0 end) as INT) as '0906',
                cast((case when mocodes like '%0907%' then 1 else 0 end) as INT) as '0907',
                cast((case when mocodes like '%0908%' then 1 else 0 end) as INT) as '0908',
                cast((case when mocodes like '%0909%' then 1 else 0 end) as INT) as '0909',
                cast((case when mocodes like '%0910%' then 1 else 0 end) as INT) as '0910',
                cast((case when mocodes like '%0911%' then 1 else 0 end) as INT) as '0911',
                cast((case when mocodes like '%0912%' then 1 else 0 end) as INT) as '0912',
                cast((case when mocodes like '%0913%' then 1 else 0 end) as INT) as '0913',
                cast((case when mocodes like '%0914%' then 1 else 0 end) as INT) as '0914',
                cast((case when mocodes like '%0915%' then 1 else 0 end) as INT) as '0915',
                cast((case when mocodes like '%0916%' then 1 else 0 end) as INT) as '0916',
                cast((case when mocodes like '%0917%' then 1 else 0 end) as INT) as '0917',
                cast((case when mocodes like '%0918%' then 1 else 0 end) as INT) as '0918',
                cast((case when mocodes like '%0919%' then 1 else 0 end) as INT) as '0919',
                cast((case when mocodes like '%0920%' then 1 else 0 end) as INT) as '0920',
                cast((case when mocodes like '%0921%' then 1 else 0 end) as INT) as '0921',
                cast((case when mocodes like '%0922%' then 1 else 0 end) as INT) as '0922',
                cast((case when mocodes like '%0923%' then 1 else 0 end) as INT) as '0923',
                cast((case when mocodes like '%0924%' then 1 else 0 end) as INT) as '0924',
                cast((case when mocodes like '%0925%' then 1 else 0 end) as INT) as '0925',
                cast((case when mocodes like '%0926%' then 1 else 0 end) as INT) as '0926',
                cast((case when mocodes like '%0927%' then 1 else 0 end) as INT) as '0927',
                cast((case when mocodes like '%0928%' then 1 else 0 end) as INT) as '0928',
                cast((case when mocodes like '%0929%' then 1 else 0 end) as INT) as '0929',
                cast((case when mocodes like '%0930%' then 1 else 0 end) as INT) as '0930',
                cast((case when mocodes like '%0931%' then 1 else 0 end) as INT) as '0931',
                cast((case when mocodes like '%0932%' then 1 else 0 end) as INT) as '0932',
                cast((case when mocodes like '%0933%' then 1 else 0 end) as INT) as '0933',
                cast((case when mocodes like '%0934%' then 1 else 0 end) as INT) as '0934',
                cast((case when mocodes like '%0935%' then 1 else 0 end) as INT) as '0935',
                cast((case when mocodes like '%0936%' then 1 else 0 end) as INT) as '0936',
                cast((case when mocodes like '%0937%' then 1 else 0 end) as INT) as '0937',
                cast((case when mocodes like '%0938%' then 1 else 0 end) as INT) as '0938',
                cast((case when mocodes like '%0939%' then 1 else 0 end) as INT) as '0939',
                cast((case when mocodes like '%0940%' then 1 else 0 end) as INT) as '0940',
                cast((case when mocodes like '%0941%' then 1 else 0 end) as INT) as '0941',
                cast((case when mocodes like '%0942%' then 1 else 0 end) as INT) as '0942',
                cast((case when mocodes like '%0943%' then 1 else 0 end) as INT) as '0943',
                cast((case when mocodes like '%0944%' then 1 else 0 end) as INT) as '0944',
                cast((case when mocodes like '%0945%' then 1 else 0 end) as INT) as '0945',
                cast((case when mocodes like '%0946%' then 1 else 0 end) as INT) as '0946',
                cast((case when mocodes like '%1000%' then 1 else 0 end) as INT) as '1000',
                cast((case when mocodes like '%1001%' then 1 else 0 end) as INT) as '1001',
                cast((case when mocodes like '%1002%' then 1 else 0 end) as INT) as '1002',
                cast((case when mocodes like '%1003%' then 1 else 0 end) as INT) as '1003',
                cast((case when mocodes like '%1004%' then 1 else 0 end) as INT) as '1004',
                cast((case when mocodes like '%1005%' then 1 else 0 end) as INT) as '1005',
                cast((case when mocodes like '%1006%' then 1 else 0 end) as INT) as '1006',
                cast((case when mocodes like '%1007%' then 1 else 0 end) as INT) as '1007',
                cast((case when mocodes like '%1008%' then 1 else 0 end) as INT) as '1008',
                cast((case when mocodes like '%1009%' then 1 else 0 end) as INT) as '1009',
                cast((case when mocodes like '%1010%' then 1 else 0 end) as INT) as '1010',
                cast((case when mocodes like '%1011%' then 1 else 0 end) as INT) as '1011',
                cast((case when mocodes like '%1012%' then 1 else 0 end) as INT) as '1012',
                cast((case when mocodes like '%1013%' then 1 else 0 end) as INT) as '1013',
                cast((case when mocodes like '%1014%' then 1 else 0 end) as INT) as '1014',
                cast((case when mocodes like '%1015%' then 1 else 0 end) as INT) as '1015',
                cast((case when mocodes like '%1016%' then 1 else 0 end) as INT) as '1016',
                cast((case when mocodes like '%1017%' then 1 else 0 end) as INT) as '1017',
                cast((case when mocodes like '%1018%' then 1 else 0 end) as INT) as '1018',
                cast((case when mocodes like '%1019%' then 1 else 0 end) as INT) as '1019',
                cast((case when mocodes like '%1020%' then 1 else 0 end) as INT) as '1020',
                cast((case when mocodes like '%1021%' then 1 else 0 end) as INT) as '1021',
                cast((case when mocodes like '%1022%' then 1 else 0 end) as INT) as '1022',
                cast((case when mocodes like '%1023%' then 1 else 0 end) as INT) as '1023',
                cast((case when mocodes like '%1024%' then 1 else 0 end) as INT) as '1024',
                cast((case when mocodes like '%1025%' then 1 else 0 end) as INT) as '1025',
                cast((case when mocodes like '%1026%' then 1 else 0 end) as INT) as '1026',
                cast((case when mocodes like '%1027%' then 1 else 0 end) as INT) as '1027',
                cast((case when mocodes like '%1028%' then 1 else 0 end) as INT) as '1028',
                cast((case when mocodes like '%1100%' then 1 else 0 end) as INT) as '1100',
                cast((case when mocodes like '%1101%' then 1 else 0 end) as INT) as '1101',
                cast((case when mocodes like '%1201%' then 1 else 0 end) as INT) as '1201',
                cast((case when mocodes like '%1202%' then 1 else 0 end) as INT) as '1202',
                cast((case when mocodes like '%1203%' then 1 else 0 end) as INT) as '1203',
                cast((case when mocodes like '%1204%' then 1 else 0 end) as INT) as '1204',
                cast((case when mocodes like '%1205%' then 1 else 0 end) as INT) as '1205',
                cast((case when mocodes like '%1206%' then 1 else 0 end) as INT) as '1206',
                cast((case when mocodes like '%1207%' then 1 else 0 end) as INT) as '1207',
                cast((case when mocodes like '%1208%' then 1 else 0 end) as INT) as '1208',
                cast((case when mocodes like '%1209%' then 1 else 0 end) as INT) as '1209',
                cast((case when mocodes like '%1210%' then 1 else 0 end) as INT) as '1210',
                cast((case when mocodes like '%1211%' then 1 else 0 end) as INT) as '1211',
                cast((case when mocodes like '%1212%' then 1 else 0 end) as INT) as '1212',
                cast((case when mocodes like '%1213%' then 1 else 0 end) as INT) as '1213',
                cast((case when mocodes like '%1214%' then 1 else 0 end) as INT) as '1214',
                cast((case when mocodes like '%1215%' then 1 else 0 end) as INT) as '1215',
                cast((case when mocodes like '%1216%' then 1 else 0 end) as INT) as '1216',
                cast((case when mocodes like '%1217%' then 1 else 0 end) as INT) as '1217',
                cast((case when mocodes like '%1218%' then 1 else 0 end) as INT) as '1218',
                cast((case when mocodes like '%1219%' then 1 else 0 end) as INT) as '1219',
                cast((case when mocodes like '%1220%' then 1 else 0 end) as INT) as '1220',
                cast((case when mocodes like '%1221%' then 1 else 0 end) as INT) as '1221',
                cast((case when mocodes like '%1222%' then 1 else 0 end) as INT) as '1222',
                cast((case when mocodes like '%1223%' then 1 else 0 end) as INT) as '1223',
                cast((case when mocodes like '%1224%' then 1 else 0 end) as INT) as '1224',
                cast((case when mocodes like '%1225%' then 1 else 0 end) as INT) as '1225',
                cast((case when mocodes like '%1226%' then 1 else 0 end) as INT) as '1226',
                cast((case when mocodes like '%1227%' then 1 else 0 end) as INT) as '1227',
                cast((case when mocodes like '%1228%' then 1 else 0 end) as INT) as '1228',
                cast((case when mocodes like '%1229%' then 1 else 0 end) as INT) as '1229',
                cast((case when mocodes like '%1230%' then 1 else 0 end) as INT) as '1230',
                cast((case when mocodes like '%1231%' then 1 else 0 end) as INT) as '1231',
                cast((case when mocodes like '%1232%' then 1 else 0 end) as INT) as '1232',
                cast((case when mocodes like '%1233%' then 1 else 0 end) as INT) as '1233',
                cast((case when mocodes like '%1234%' then 1 else 0 end) as INT) as '1234',
                cast((case when mocodes like '%1235%' then 1 else 0 end) as INT) as '1235',
                cast((case when mocodes like '%1236%' then 1 else 0 end) as INT) as '1236',
                cast((case when mocodes like '%1237%' then 1 else 0 end) as INT) as '1237',
                cast((case when mocodes like '%1238%' then 1 else 0 end) as INT) as '1238',
                cast((case when mocodes like '%1239%' then 1 else 0 end) as INT) as '1239',
                cast((case when mocodes like '%1240%' then 1 else 0 end) as INT) as '1240',
                cast((case when mocodes like '%1241%' then 1 else 0 end) as INT) as '1241',
                cast((case when mocodes like '%1242%' then 1 else 0 end) as INT) as '1242',
                cast((case when mocodes like '%1243%' then 1 else 0 end) as INT) as '1243',
                cast((case when mocodes like '%1244%' then 1 else 0 end) as INT) as '1244',
                cast((case when mocodes like '%1245%' then 1 else 0 end) as INT) as '1245',
                cast((case when mocodes like '%1247%' then 1 else 0 end) as INT) as '1247',
                cast((case when mocodes like '%1248%' then 1 else 0 end) as INT) as '1248',
                cast((case when mocodes like '%1251%' then 1 else 0 end) as INT) as '1251',
                cast((case when mocodes like '%1252%' then 1 else 0 end) as INT) as '1252',
                cast((case when mocodes like '%1253%' then 1 else 0 end) as INT) as '1253',
                cast((case when mocodes like '%1254%' then 1 else 0 end) as INT) as '1254',
                cast((case when mocodes like '%1255%' then 1 else 0 end) as INT) as '1255',
                cast((case when mocodes like '%1256%' then 1 else 0 end) as INT) as '1256',
                cast((case when mocodes like '%1257%' then 1 else 0 end) as INT) as '1257',
                cast((case when mocodes like '%1258%' then 1 else 0 end) as INT) as '1258',
                cast((case when mocodes like '%1259%' then 1 else 0 end) as INT) as '1259',
                cast((case when mocodes like '%1260%' then 1 else 0 end) as INT) as '1260',
                cast((case when mocodes like '%1261%' then 1 else 0 end) as INT) as '1261',
                cast((case when mocodes like '%1262%' then 1 else 0 end) as INT) as '1262',
                cast((case when mocodes like '%1263%' then 1 else 0 end) as INT) as '1263',
                cast((case when mocodes like '%1264%' then 1 else 0 end) as INT) as '1264',
                cast((case when mocodes like '%1265%' then 1 else 0 end) as INT) as '1265',
                cast((case when mocodes like '%1266%' then 1 else 0 end) as INT) as '1266',
                cast((case when mocodes like '%1267%' then 1 else 0 end) as INT) as '1267',
                cast((case when mocodes like '%1268%' then 1 else 0 end) as INT) as '1268',
                cast((case when mocodes like '%1269%' then 1 else 0 end) as INT) as '1269',
                cast((case when mocodes like '%1270%' then 1 else 0 end) as INT) as '1270',
                cast((case when mocodes like '%1271%' then 1 else 0 end) as INT) as '1271',
                cast((case when mocodes like '%1272%' then 1 else 0 end) as INT) as '1272',
                cast((case when mocodes like '%1273%' then 1 else 0 end) as INT) as '1273',
                cast((case when mocodes like '%1274%' then 1 else 0 end) as INT) as '1274',
                cast((case when mocodes like '%1275%' then 1 else 0 end) as INT) as '1275',
                cast((case when mocodes like '%1276%' then 1 else 0 end) as INT) as '1276',
                cast((case when mocodes like '%1277%' then 1 else 0 end) as INT) as '1277',
                cast((case when mocodes like '%1278%' then 1 else 0 end) as INT) as '1278',
                cast((case when mocodes like '%1279%' then 1 else 0 end) as INT) as '1279',
                cast((case when mocodes like '%1280%' then 1 else 0 end) as INT) as '1280',
                cast((case when mocodes like '%1281%' then 1 else 0 end) as INT) as '1281',
                cast((case when mocodes like '%1300%' then 1 else 0 end) as INT) as '1300',
                cast((case when mocodes like '%1301%' then 1 else 0 end) as INT) as '1301',
                cast((case when mocodes like '%1302%' then 1 else 0 end) as INT) as '1302',
                cast((case when mocodes like '%1303%' then 1 else 0 end) as INT) as '1303',
                cast((case when mocodes like '%1304%' then 1 else 0 end) as INT) as '1304',
                cast((case when mocodes like '%1305%' then 1 else 0 end) as INT) as '1305',
                cast((case when mocodes like '%1306%' then 1 else 0 end) as INT) as '1306',
                cast((case when mocodes like '%1307%' then 1 else 0 end) as INT) as '1307',
                cast((case when mocodes like '%1308%' then 1 else 0 end) as INT) as '1308',
                cast((case when mocodes like '%1309%' then 1 else 0 end) as INT) as '1309',
                cast((case when mocodes like '%1310%' then 1 else 0 end) as INT) as '1310',
                cast((case when mocodes like '%1311%' then 1 else 0 end) as INT) as '1311',
                cast((case when mocodes like '%1312%' then 1 else 0 end) as INT) as '1312',
                cast((case when mocodes like '%1313%' then 1 else 0 end) as INT) as '1313',
                cast((case when mocodes like '%1314%' then 1 else 0 end) as INT) as '1314',
                cast((case when mocodes like '%1315%' then 1 else 0 end) as INT) as '1315',
                cast((case when mocodes like '%1316%' then 1 else 0 end) as INT) as '1316',
                cast((case when mocodes like '%1317%' then 1 else 0 end) as INT) as '1317',
                cast((case when mocodes like '%1318%' then 1 else 0 end) as INT) as '1318',
                cast((case when mocodes like '%1401%' then 1 else 0 end) as INT) as '1401',
                cast((case when mocodes like '%1402%' then 1 else 0 end) as INT) as '1402',
                cast((case when mocodes like '%1403%' then 1 else 0 end) as INT) as '1403',
                cast((case when mocodes like '%1404%' then 1 else 0 end) as INT) as '1404',
                cast((case when mocodes like '%1405%' then 1 else 0 end) as INT) as '1405',
                cast((case when mocodes like '%1406%' then 1 else 0 end) as INT) as '1406',
                cast((case when mocodes like '%1407%' then 1 else 0 end) as INT) as '1407',
                cast((case when mocodes like '%1408%' then 1 else 0 end) as INT) as '1408',
                cast((case when mocodes like '%1409%' then 1 else 0 end) as INT) as '1409',
                cast((case when mocodes like '%1410%' then 1 else 0 end) as INT) as '1410',
                cast((case when mocodes like '%1411%' then 1 else 0 end) as INT) as '1411',
                cast((case when mocodes like '%1412%' then 1 else 0 end) as INT) as '1412',
                cast((case when mocodes like '%1413%' then 1 else 0 end) as INT) as '1413',
                cast((case when mocodes like '%1414%' then 1 else 0 end) as INT) as '1414',
                cast((case when mocodes like '%1415%' then 1 else 0 end) as INT) as '1415',
                cast((case when mocodes like '%1416%' then 1 else 0 end) as INT) as '1416',
                cast((case when mocodes like '%1417%' then 1 else 0 end) as INT) as '1417',
                cast((case when mocodes like '%1418%' then 1 else 0 end) as INT) as '1418',
                cast((case when mocodes like '%1419%' then 1 else 0 end) as INT) as '1419',
                cast((case when mocodes like '%1420%' then 1 else 0 end) as INT) as '1420',
                cast((case when mocodes like '%1501%' then 1 else 0 end) as INT) as '1501',
                cast((case when mocodes like '%1601%' then 1 else 0 end) as INT) as '1601',
                cast((case when mocodes like '%1602%' then 1 else 0 end) as INT) as '1602',
                cast((case when mocodes like '%1603%' then 1 else 0 end) as INT) as '1603',
                cast((case when mocodes like '%1604%' then 1 else 0 end) as INT) as '1604',
                cast((case when mocodes like '%1605%' then 1 else 0 end) as INT) as '1605',
                cast((case when mocodes like '%1606%' then 1 else 0 end) as INT) as '1606',
                cast((case when mocodes like '%1607%' then 1 else 0 end) as INT) as '1607',
                cast((case when mocodes like '%1608%' then 1 else 0 end) as INT) as '1608',
                cast((case when mocodes like '%1609%' then 1 else 0 end) as INT) as '1609',
                cast((case when mocodes like '%1610%' then 1 else 0 end) as INT) as '1610',
                cast((case when mocodes like '%1611%' then 1 else 0 end) as INT) as '1611',
                cast((case when mocodes like '%1612%' then 1 else 0 end) as INT) as '1612',
                cast((case when mocodes like '%1701%' then 1 else 0 end) as INT) as '1701',
                cast((case when mocodes like '%1702%' then 1 else 0 end) as INT) as '1702',
                cast((case when mocodes like '%1801%' then 1 else 0 end) as INT) as '1801',
                cast((case when mocodes like '%1802%' then 1 else 0 end) as INT) as '1802',
                cast((case when mocodes like '%1803%' then 1 else 0 end) as INT) as '1803',
                cast((case when mocodes like '%1804%' then 1 else 0 end) as INT) as '1804',
                cast((case when mocodes like '%1805%' then 1 else 0 end) as INT) as '1805',
                cast((case when mocodes like '%1806%' then 1 else 0 end) as INT) as '1806',
                cast((case when mocodes like '%1807%' then 1 else 0 end) as INT) as '1807',
                cast((case when mocodes like '%1808%' then 1 else 0 end) as INT) as '1808',
                cast((case when mocodes like '%1809%' then 1 else 0 end) as INT) as '1809',
                cast((case when mocodes like '%1810%' then 1 else 0 end) as INT) as '1810',
                cast((case when mocodes like '%1811%' then 1 else 0 end) as INT) as '1811',
                cast((case when mocodes like '%1812%' then 1 else 0 end) as INT) as '1812',
                cast((case when mocodes like '%1813%' then 1 else 0 end) as INT) as '1813',
                cast((case when mocodes like '%1814%' then 1 else 0 end) as INT) as '1814',
                cast((case when mocodes like '%1815%' then 1 else 0 end) as INT) as '1815',
                cast((case when mocodes like '%1816%' then 1 else 0 end) as INT) as '1816',
                cast((case when mocodes like '%1817%' then 1 else 0 end) as INT) as '1817',
                cast((case when mocodes like '%1818%' then 1 else 0 end) as INT) as '1818',
                cast((case when mocodes like '%1819%' then 1 else 0 end) as INT) as '1819',
                cast((case when mocodes like '%1820%' then 1 else 0 end) as INT) as '1820',
                cast((case when mocodes like '%1821%' then 1 else 0 end) as INT) as '1821',
                cast((case when mocodes like '%1822%' then 1 else 0 end) as INT) as '1822',
                cast((case when mocodes like '%1823%' then 1 else 0 end) as INT) as '1823',
                cast((case when mocodes like '%1824%' then 1 else 0 end) as INT) as '1824',
                cast((case when mocodes like '%1900%' then 1 else 0 end) as INT) as '1900',
                cast((case when mocodes like '%1901%' then 1 else 0 end) as INT) as '1901',
                cast((case when mocodes like '%1902%' then 1 else 0 end) as INT) as '1902',
                cast((case when mocodes like '%1903%' then 1 else 0 end) as INT) as '1903',
                cast((case when mocodes like '%1904%' then 1 else 0 end) as INT) as '1904',
                cast((case when mocodes like '%1905%' then 1 else 0 end) as INT) as '1905',
                cast((case when mocodes like '%1906%' then 1 else 0 end) as INT) as '1906',
                cast((case when mocodes like '%1907%' then 1 else 0 end) as INT) as '1907',
                cast((case when mocodes like '%1908%' then 1 else 0 end) as INT) as '1908',
                cast((case when mocodes like '%1909%' then 1 else 0 end) as INT) as '1909',
                cast((case when mocodes like '%1910%' then 1 else 0 end) as INT) as '1910',
                cast((case when mocodes like '%1911%' then 1 else 0 end) as INT) as '1911',
                cast((case when mocodes like '%1912%' then 1 else 0 end) as INT) as '1912',
                cast((case when mocodes like '%1913%' then 1 else 0 end) as INT) as '1913',
                cast((case when mocodes like '%1914%' then 1 else 0 end) as INT) as '1914',
                cast((case when mocodes like '%1915%' then 1 else 0 end) as INT) as '1915',
                cast((case when mocodes like '%1916%' then 1 else 0 end) as INT) as '1916',
                cast((case when mocodes like '%2000%' then 1 else 0 end) as INT) as '2000',
                cast((case when mocodes like '%2001%' then 1 else 0 end) as INT) as '2001',
                cast((case when mocodes like '%2002%' then 1 else 0 end) as INT) as '2002',
                cast((case when mocodes like '%2003%' then 1 else 0 end) as INT) as '2003',
                cast((case when mocodes like '%2004%' then 1 else 0 end) as INT) as '2004',
                cast((case when mocodes like '%2005%' then 1 else 0 end) as INT) as '2005',
                cast((case when mocodes like '%2006%' then 1 else 0 end) as INT) as '2006',
                cast((case when mocodes like '%2007%' then 1 else 0 end) as INT) as '2007',
                cast((case when mocodes like '%2008%' then 1 else 0 end) as INT) as '2008',
                cast((case when mocodes like '%2009%' then 1 else 0 end) as INT) as '2009',
                cast((case when mocodes like '%2010%' then 1 else 0 end) as INT) as '2010',
                cast((case when mocodes like '%2011%' then 1 else 0 end) as INT) as '2011',
                cast((case when mocodes like '%2012%' then 1 else 0 end) as INT) as '2012',
                cast((case when mocodes like '%2013%' then 1 else 0 end) as INT) as '2013',
                cast((case when mocodes like '%2014%' then 1 else 0 end) as INT) as '2014',
                cast((case when mocodes like '%2015%' then 1 else 0 end) as INT) as '2015',
                cast((case when mocodes like '%9999%' then 1 else 0 end) as INT) as '9999'
                from la_crime_incl_fbi_cat 
                ''')
    conn.commit()
    conn.close()

    fbi_8_crime_cat = pd.read_sql('''select * from new_ml_8_cat_crime_data''',con=sqlite3.connect("db/la_crime.db"))
    fbi_crime_cln = fbi_8_crime_cat.dropna()
    
    #fbi_crime_cln = pd.read_csv("fbi_crime_clean.csv")

    X = fbi_crime_cln.drop("FBI_Category", axis=1)
    y = fbi_crime_cln["FBI_Category"]

    #print("printing X,y")
    #print(X, y)
    
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from keras.utils import to_categorical

    X_scaler = StandardScaler().fit(X)
    X_scaled = X_scaler.transform(X)

    # Step 1: Label-encode data set
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    encoded_y = label_encoder.transform(y)

    # Step 2: Convert encoded labels to one-hot-encoding
    y_categorical = to_categorical(encoded_y)

    # Load the DeepNN_model
    #from keras.models import load_model
    #crime_model = load_model("ml_model/DeepNN_model/fbi_category_model_trained.h5")
    #model_loss, model_accuracy = crime_model.evaluate(
    #X_scaled, y_categorical, verbose=2)

    # Make FBI crime category predictions using trained model


    global graph
    with graph.as_default():
        encoded_predictions = crime_model.predict_classes(X_scaled)
        prediction_labels = label_encoder.inverse_transform(encoded_predictions)
    
    dr_no = fbi_crime_cln['dr_no'].tolist()

    # Create dictionary and populate it wth dr_no (key) and prediction_labels (values)
    crime_predictions = {}

    for dr, prediction_label in zip(dr_no, prediction_labels):
        crime_predictions.update({str(dr): str(prediction_label)})

    # Read dictionary into dataframe
    crime_8_cat_predictions = pd.DataFrame.from_dict(crime_predictions, orient='index')
    crime_8_cat_predictions = crime_8_cat_predictions.reset_index().rename(columns={0:'FBI_Cat_Prediction','index':'dr_no'})

    # Create FBI_Cat_Prediction table
    from sqlalchemy import create_engine
    engine = create_engine('sqlite:///db/la_crime.db')
    crime_8_cat_predictions.to_sql('fbi_cat_prediction', engine, if_exists='replace', index=True, )

    # Create new_la_crime_incl_predictions table
    conn = sqlite3.connect("db/la_crime.db")
    c = conn.cursor()
    c.execute('''drop table if exists new_la_crime_incl_predictions;''')
    c.execute('''
                    create table new_la_crime_incl_predictions as select
                    la.dr_no,
                    area_id,
                    area_name,
                    crm_cd,
                    crm_cd_desc,
                    cross_street,
                    date_occ,
                    date_rptd,
                    location,
                    longitude,
                    latitude,
                    mocodes,
                    premis_cd,
                    premis_desc,
                    rpt_dist_no,
                    status,
                    status_desc,
                    hour_occ,
                    minute_occ,
                    vict_age,
                    vict_descent,
                    vict_sex,
                    weapon_desc,
                    weapon_used_cd,
                    loc_type,
                    FBI_Category,
                    FBI_Cat_Prediction
                    from la_crime_incl_fbi_cat as la
                    inner join fbi_cat_prediction as pr
                    on la.dr_no = pr.dr_no
                    group by 
                    la.dr_no,
                    area_id,
                    area_name,
                    crm_cd,
                    crm_cd_desc,
                    cross_street,
                    date_occ,
                    date_rptd,
                    location,
                    longitude,
                    latitude,
                    mocodes,
                    premis_cd,
                    premis_desc,
                    rpt_dist_no,
                    status,
                    status_desc,
                    hour_occ,
                    minute_occ,
                    vict_age,
                    vict_descent,
                    vict_sex,
                    weapon_desc,
                    weapon_used_cd,
                    loc_type,
                    FBI_Category,
                    FBI_Cat_Prediction
                    ;  ''')
    conn.commit()
    conn.close()

    # Read new_la_crime_incl_pred into dataframe
    new_la_crime_incl_pred = pd.read_sql('''select * from new_la_crime_incl_predictions;''',con=sqlite3.connect("db/la_crime.db")).set_index('dr_no')
   
    # Replace mocodes in new_la_crime_incl_pred with mo descriptions
    new_la_crime_incl_pred['date_occ'] = new_la_crime_incl_pred['date_occ'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
    new_la_crime_incl_pred['date_rptd'] = new_la_crime_incl_pred['date_rptd'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
    new_la_crime_incl_pred['FBI_Cat_Prediction'] = new_la_crime_incl_pred['FBI_Cat_Prediction'].str.replace('BTFV', 'Burglary/Theft From Vehicle')
    new_la_crime_incl_pred['FBI_Cat_Prediction'] = new_la_crime_incl_pred['FBI_Cat_Prediction'].str.strip(" (121, 122)")
    new_la_crime_incl_pred['FBI_Category'] = new_la_crime_incl_pred['FBI_Category'].str.replace('BTFV', 'Burglary/Theft From Vehicle')
    new_la_crime_incl_pred['FBI_Category'] = new_la_crime_incl_pred['FBI_Category'].str.strip(" (121, 122)")
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].fillna('["No MO codes on this report"]')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0100', 'Suspect Impersonate')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0101', 'Aid victim')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0102', 'Blind')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0103', 'Crippled')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0104', 'Customer')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0105', 'Delivery')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0106', 'Doctor')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0107', 'God')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0108', 'Infirm')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0109', 'Inspector')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0110', 'Involved in traffic/accident')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0112', 'Police')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0113', 'Renting')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0114', 'Repair Person')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0115', 'Returning stolen property')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0116', 'Satan')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0117', 'Salesman')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0118', 'Seeking someone')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0119', 'Sent by owner')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0120', 'Social Security/Medicare')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0121', 'DWP/Gas Company/Utility worker')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0122', 'Contractor')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0123', 'Gardener/Tree Trimmer')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0200', 'Suspect wore disguise')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0201', 'Bag')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0202', 'Cap/hat')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0203', 'Cloth (with eyeholes)')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0204', 'Clothes of opposite sex')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0205', 'Earring')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0206', 'Gloves')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0207', 'Handkerchief')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0208', 'Halloween mask')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0209', 'Mask')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0210', 'Make up (males only)')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0211', 'Shoes')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0212', 'Nude/partly nude')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0213', 'Ski mask')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0214', 'Stocking')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0215', 'Unusual clothes')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0216', 'Suspect wore hood/hoodie')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0217', 'Uniform')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0218', 'Wig')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0219', 'Mustache-Fake')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0220', 'Suspect wore motorcycle helmet')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0301', 'Escaped on (used) transit train')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0302', 'Aimed gun')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0303', 'Ambushed')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0304', 'Ate/drank on premises')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0305', 'Attacks from rear')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0306', 'Crime on upper floor')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0307', 'Defecated/urinated')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0308', 'Demands jewelry')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0309', 'Drive-by shooting')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0310', 'Got victim to withdraw savings')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0311', 'Graffiti')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0312', 'Gun in waistband')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0313', 'Hid in building')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0314', 'Hot Prowl')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0315', 'Jumped counter/goes behind counter')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0316', 'Makes victim give money')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0317', 'Pillowcase/suitcase')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0318', 'Prepared exit')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0319', 'Profanity Used')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0320', 'Quiet polite')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0321', 'Ransacked')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0322', 'Smashed display case')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0323', 'Smoked on premises')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0324', 'Takes money from register')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0325', 'Took merchandise')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0326', 'Used driver')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0327', 'Used lookout')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0328', 'Used toilet')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0329', 'Vandalized')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0330', 'Victims vehicle taken')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0331', 'Mailbox Bombing')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0332', 'Mailbox Vandalism')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0333', 'Used hand held radios')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0334', 'Brandishes weapon')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0335', 'Cases location')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0336', 'Chain snatch')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0337', 'Demands money')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0338', 'Disables Telephone')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0339', 'Disables video camera')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0340', 'Suspect follows victim/follows victim home')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0341', 'Makes vict lie down')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0342', 'Multi-susps overwhelm')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0343', 'Orders vict to rear room')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0344', 'Removes vict property')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0345', 'Riding bike')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0346', 'Snatch property and runs')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0347', 'Stalks vict')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0348', 'Takeover other')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0349', 'Takes mail')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0350', 'Concealed victim\'s body')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0351', 'Disabled Security')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0352', 'Took Victim\'s clothing or jewelry')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0353', 'Weapon Concealed')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0354', 'Suspect takes car keys')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0355', 'Demanded property other than money')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0356', 'Suspect spits on victim')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0357', 'Cuts or breaks purse strap')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0358', 'Forces Entry')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0359', 'Made unusual statement')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0360', 'Suspect is Other Family Member')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0361', 'Suspect is neighbor')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0362', 'Suspect attempts to carry victim away')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0363', 'Home invasion')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0364', 'Suspect is babysitter')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0365', 'Takeover robbery')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0366', 'Ordered vict to open safe')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0367', 'Was Transit Patrol')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0368', 'Suspect speaks foreign language')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0369', 'Suspect speaks spanish')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0370', 'Frisks victim/pats down victim/searches victim')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0371', 'Gang affiliation questions asked/made gang statement')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0372', 'Photographed victim/took pictures of victim')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0373', 'Handicapped/in wheelchair')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0374', 'Gang signs/threw gang signs using hands')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0375', 'Removes cash register')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0376', 'Makes victim kneel')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0377', 'Takes vict\'s identification/driver license')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0378', 'Brings own bag')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0379', 'Turns off lights/electricity')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0380', 'Distracts Victim')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0381', 'Suspect apologizes')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0382', 'Removed money/property from safe')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0383', 'Suspect entered during open house/party/estate/yard sale')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0384', 'Suspect removed drugs from location')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0385', 'Suspect removed parts from vehicle')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0386', 'Suspect removed property from trunk of vehicle')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0387', 'Weapon (other than gun) in waistband')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0388', 'Suspect points laser at plane/helicopter')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0389', 'Knock-knock')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0390', 'Purse snatch')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0391', 'Used demand note')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0392', 'False Emergency Reporting')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0393', '911 Abuse')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0394', 'Susp takes UPS, Fedex, USPS packages')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0395', 'Murder/Suicide')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0396', 'Used paper plates to disguise license number')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0397', 'Cut lock (to bicycle, gate, etc.')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0398', 'Roof access (remove A/C, equip, etc.)')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0399', 'Vehicle to Vehicle shooting')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0400', 'Force used')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0401', 'Bit')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0402', 'Blindfolded')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0403', 'Bomb Threat, Bomb found')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0404', 'Bomb Threat, no bomb')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0405', 'Bound')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0406', 'Brutal Assault')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0407', 'Burned Victim')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0408', 'Choked/uses choke hold')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0409', 'Cover mouth w/hands')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0410', 'Covered victim\'s face')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0411', 'Cut/stabbed')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0412', 'Disfigured')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0413', 'Drugged')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0414', 'Gagged')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0415', 'Handcuffed/Metal')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0416', 'Hit-Hit w/ weapon')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0417', 'Kicked')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0418', 'Kidnapped')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0419', 'Pulled victims hair')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0420', 'Searched')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0421', 'Threaten to kill')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0422', 'Threaten Victims family')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0423', 'Tied victim to object')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0424', 'Tore clothes off victim')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0425', 'Tortured')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0426', 'Twisted arm')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0427', 'Whipped')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0428', 'Dismembered')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0429', 'Vict knocked to ground')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0430', 'Vict shot')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0431', 'Sprayed with chemical')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0432', 'Intimidation')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0433', 'Makes victim kneel')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0434', 'Bed Sheets/Linens')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0435', 'Chain')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0436', 'Clothing')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0437', 'Flexcuffs/Plastic Tie')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0438', 'Rope/Cordage')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0439', 'Tape/Electrical etc...')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0440', 'Telephone/Electric Cord')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0441', 'Wire')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0442', 'Active Shooter')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0443', 'Threaten to harm victim (other than kill)')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0444', 'Pushed')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0445', 'Suspect swung weapon')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0446', 'Suspect swung fist')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0447', 'Suspect threw object at victim')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0448', 'Grabbed')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0449', 'Put a weapon to body')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0450', 'Suspect shot at victim (no hits)')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0500', 'Sex related acts')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0501', 'Susp ejaculated outside victim')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0502', 'Fecal Fetish')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0503', 'Fondle victim')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0504', 'Forced to disrobe')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0505', 'Forced to fondle suspect')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0506', 'Forced to masturbate suspect')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0507', 'Forced to orally copulate suspect')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0508', 'Hit victim prior, during, after act')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0509', 'Hugged')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0510', 'Kissed victims body/face')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0511', 'Masochism/bondage')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0512', 'Orally copulated victim')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0513', 'Photographed victim')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0514', 'Pornography')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0515', 'Put hand, finger or object into vagina')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0516', 'Reached climax/ejaculated')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0517', 'Sadism')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0518', 'Simulated intercourse')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0519', 'Sodomy')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0520', 'Solicited/offered immoral act')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0521', 'Tongue or mouth to anus')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0522', 'Touched')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0523', 'Unable to get erection')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0524', 'Underwear Fetish')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0525', 'Urinated')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0526', 'Utilized Condom')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0527', 'Actual Intercourse')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0528', 'Masturbate')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0529', 'Indecent Exposure')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0530', 'Used lubricant')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0531', 'Suspect made sexually suggestive remarks')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0532', 'Suspect undressed victim')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0533', 'Consensual Sex')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0534', 'Suspect in vehicle nude/partially nude')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0535', 'Suspect asks minor\'s name')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0536', 'Suspect removes own clothing')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0537', 'Suspect removes victim\'s clothing')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0538', 'Suspect fondles self')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0539', 'Suspect puts hand in victim\'s rectum')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0540', 'Suspect puts finger(s) in victim\'s rectum')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0541', 'Suspect puts object(s) in victim\'s rectum')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0542', 'Orders victim to undress')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0543', 'Orders victim to fondle suspect')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0544', 'Orders victim to fondle self')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0545', 'Male Victim of sexual assault')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0546', 'Susp instructs vict to make certain statements')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0547', 'Suspect force vict to bathe/clean/wipe')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0548', 'Suspect gives victim douche/enema')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0549', 'Suspect ejaculates in victims mouth')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0550', 'Suspect licks victim')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0551', 'Suspect touches victim genitalia/genitals over clothing')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0552', 'Suspect is Victim\'s Father')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0553', 'Suspect is Victim\'s Mother')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0554', 'Suspect is Victim\'s Brother')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0555', 'Suspect is Victim\'s Sister')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0556', 'Suspect is Victim\'s Step-Father')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0557', 'Suspect is Victim\'s Step-Mother')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0558', 'Suspect is Victim\'s Uncle')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0559', 'Suspect is Victim\'s Aunt')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0560', 'Suspect is Victim\'s Guardian')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0561', 'Suspect is Victim\'s Son')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0562', 'Suspect is Victim\'s Daughter')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0563', 'Fetish, Other')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0601', 'Business')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0602', 'Family')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0603', 'Landlord/Tenant/Neighbor')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0604', 'Reproductive Health Services/Facilities')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0605', 'Traffic Accident/Traffic related incident')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0701', 'THEFT: Trick or Device')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0800', 'BUNCO')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0901', 'Organized Crime')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0902', 'Political Activity')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0903', 'Hatred/Prejudice')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0904', 'Strike/Labor Troubles')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0905', 'Terrorist Group')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0906', 'Gangs')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0907', 'Narcotics (Buy-Sell-Rip)')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0908', 'Prostitution')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0909', 'Ritual/Occult')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0910', 'Public Transit')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0911', 'Revenge')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0912', 'Insurance')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0913', 'Victim knew Suspect')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0914', 'Other Felony')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0915', 'Parolee')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0916', 'Forced theft of vehicle (Car-Jacking)')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0917', 'Victim\'s Employment')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0918', 'Career Criminal')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0919', 'Road Rage')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0920', 'Homeland Security')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0921', 'Hate Incident')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0922', 'ATM Theft with PIN number')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0923', 'Stolen/Forged Checks (Personal Checks)')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0924', 'Stolen/Forged Checks (Business Checks)')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0925', 'Stolen/Forged Checks (Cashier\'s Checks)')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0926', 'Forged or Telephonic Prescription')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0927', 'Fraudulent or forged school loan')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0928', 'Forged or Fraudulent credit applications')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0929', 'Unauthorized use of victim\'s bank account information')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0930', 'Unauthorized use of victim\'s credit/debit card or number')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0931', 'Counterfeit or forged real estate documents')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0932', 'Suspect uses victim\'s identity in reporting a traffic collision')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0933', 'Suspect uses victim\'s identity when arrested')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0934', 'Suspect uses victim\'s identity when receiving a citation')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0935', 'Misc. Stolen/Forged documents')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0936', 'Dog Fighting')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0937', 'Cock Fighting')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0938', 'Animal Neglect')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0939', 'Animal Hoarding')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0940', 'Met online/Chat Room/on Party Line')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0941', 'Non-Revocable Parole (NRP)')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0942', 'Party/Flier party/Rave Party')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0943', 'Human Trafficking')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0944', 'Bait Operation')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0945', 'Estes Robbery')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('0946', 'Gang Feud')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1000', 'Suspects offers/solicits')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1001', 'Aid for vehicle')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1002', 'Amusement')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1003', 'appraise')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1004', 'Assistant')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1005', 'Audition')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1006', 'Bless')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1007', 'Candy')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1008', 'Cigarette')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1009', 'Directions')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1010', 'Drink (not liquor)')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1011', 'Employment')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1012', 'Find a job')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1013', 'Food')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1014', 'Game')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1015', 'Gift')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1016', 'Hold for safekeeping')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1017', 'Information')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1018', 'Liquor')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1019', 'Money')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1020', 'Narcotics')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1021', 'Repair')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1022', 'Ride')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1023', 'Subscriptions')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1024', 'Teach')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1025', 'Train')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1026', 'Use the phone or toilet')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1027', 'Change')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1028', 'Suspect solicits time of day')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1100', 'Shots Fired')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1101', 'Shots Fired (Animal) - Animal Services')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1201', 'Absent-advertised in paper')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1202', 'Aged (60 & over) or blind/crippled/unable to care for self')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1203', 'Victim of crime past 12 months')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1204', 'Moving')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1205', 'On Vacation/Tourist')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1206', 'Under influence drugs/liquor')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1207', 'Hitchhiker')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1208', 'Illegal Alien')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1209', 'Salesman, Jewelry')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1210', 'Professional (doctor, Lawyer, etc.)')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1211', 'Public Official')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1212', 'LA Police Officer')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1213', 'LA Fireman')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1214', 'Banking, ATM')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1215', 'Prostitute')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1216', 'Sales')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1217', 'Teenager(Use if victim\'s age is unknown)')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1218', 'Victim was Homeless/Transient')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1219', 'Nude')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1220', 'Partially Nude')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1221', 'Missing Clothing/Jewelry')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1222', 'Homosexual/Gay')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1223', 'Riding bike')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1224', 'Drive-through (not merchant)')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1225', 'Stop sign/light')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1226', 'Catering Truck Operator')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1227', 'Delivery person')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1228', 'Leaving Business Area')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1229', 'Making bank drop')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1230', 'Postal employee')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1231', 'Taxi Driver')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1232', 'Bank, Arriving at')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1233', 'Bank, Leaving')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1234', 'Bar Customer')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1235', 'Bisexual/sexually oriented towards both sexes')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1236', 'Clerk/Employer/Owner')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1237', 'Customer')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1238', 'Handicapped')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1239', 'Transgender')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1240', 'Vehicle occupant/Passenger')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1241', 'Spouse')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1242', 'Parent')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1243', 'Co-habitants')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1244', 'Victim was forced into business')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1245', 'Victim was forced into residence')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1247', 'Opening business')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1248', 'Closing business')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1251', 'Victim was a student')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1252', 'Victim was a street vendor')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1253', 'Bus Driver')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1254', 'Train Operator')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1255', 'Followed Transit System')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1256', 'Patron')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1257', 'Victim is Newborn-5 years old')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1258', 'Victim is 6 years old thru 13 years old')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1259', 'Victim is 14 years old thru 17 years old')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1260', 'Deaf/Hearing Impaired')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1261', 'Mentally Challenged/Retarded/Intellectually Slow')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1262', 'Raped while unconscious')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1263', 'Agricultural Target')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1264', 'Pipeline')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1265', 'Mailbox')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1266', 'Victim was security guard')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1267', 'Home under construction')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1268', 'Victim was 5150/Mental Illness')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1269', 'Victim was armored car driver')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1270', 'Victim was gang member')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1271', 'Victim was Law Enforcement (not LAPD)')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1272', 'Victim as at/leaving medical marijuana clinic')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1273', 'Home was being fumigated')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1274', 'Victim was Inmate/Incarcerated')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1275', 'Vacant Residence/Building')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1276', 'Pregnant')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1277', 'Gardner')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1278', 'Victim was Uber/Lyft driver')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1279', 'Victim was Foster child')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1280', 'Victim was Foster parent')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1281', 'Victim was Pistol-whipped')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1300', 'Vehicle involved')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1301', 'Forced victim vehicle to curb')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1302', 'Suspect forced way into victim\'s vehicle')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1303', 'Hid in rear seat')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1304', 'Stopped victim vehicle by flagging down, forcing T/A, etc.')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1305', 'Victim forced into vehicle')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1306', 'Victim parking, garaging vehicle')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1307', 'Breaks window')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1308', 'Drives by and snatches property')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1309', 'Susp uses vehicle')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1310', 'Victim in vehicle')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1311', 'Victim removed from vehicle')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1312', 'Suspect follows victim in vehicle')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1313', 'Suspect exits vehicle and attacks pedestrian')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1314', 'Victim loading vehicle')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1315', 'Victim unloading vehicle')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1316', 'Victim entering their vehicle')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1317', 'Victim exiting their vehicle')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1318', 'Suspect follows victim home')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1401', 'Blood Stains')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1402', 'Evidence Booked (any crime)')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1403', 'Fingerprints')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1404', 'Footprints')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1405', 'Left Note')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1406', 'Tool Marks')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1407', 'Bullets/Casings')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1408', 'Bite Marks')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1409', 'Clothes')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1410', 'Gun Shot Residue')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1411', 'Hair')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1412', 'Jewelry')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1413', 'Paint')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1414', 'Photographs')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1415', 'Rape Kit')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1416', 'Saliva')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1417', 'Semen')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1418', 'Skeleton/Bones')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1419', 'Firearm booked as evidence')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1420', 'Video surveillance booked/available')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1501', 'Other MO (see rpt)')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1601', 'Bodily Force')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1602', 'Cutting Tool')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1603', 'Knob Twist')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1604', 'Lock Box')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1605', 'Lock slip/key/pick')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1606', 'Open/unlocked')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1607', 'Pried')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1608', 'Removed')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1609', 'Smashed')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1610', 'Tunneled')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1611', 'Shaved Key')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1612', 'Punched/Pulled Door Lock')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1701', 'Elder Abuse/Physical')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1702', 'Elder Abuse/Financial')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1801', 'Susp is/was mother\'s boyfriend')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1802', 'Susp is/was victim\'s co-worker')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1803', 'Susp is/was victim\'s employee')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1804', 'Susp is/was victim\'s employer')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1805', 'Susp is/was fellow gang member')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1806', 'Susp is/was father\'s girlfriend')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1807', 'Susp is/was priest/pastor')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1808', 'Susp is/was other religious confidant')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1809', 'Susp is/was rival gang member')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1810', 'Susp is/was roommate')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1811', 'Susp is/was victim\'s teacher/coach')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1812', 'Susp is/was foster parent/sibling')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1813', 'Susp is/was current/former spouse/co-habitant')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1814', 'Susp is/was current/former boyfriend/girlfriend')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1815', 'Susp was student')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1816', 'Suspect is/was known gang member')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1817', 'Acquaintance')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1818', 'Caretaker/care-giver/nanny')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1819', 'Common-law Spouse')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1820', 'Friend')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1821', 'Spouse')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1822', 'Stranger')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1823', 'Brief encounter/Date')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1824', 'Classmate')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1900', 'Auction Fraud/eBay/cragslist,etc. (Internet based theft)')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1901', 'Child Pornography/In possession of/Via computer')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1902', 'Credit Card Fraud/Theft of services via internet')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1903', 'Cyberstalking (Stalking using internet to commit the crime)')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1904', 'Denial of computer services')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1905', 'Destruction of computer data')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1906', 'Harrassing E-Mail/Text Message/Other Electronic Communications')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1907', 'Hate Crime materials/printouts/e-mails')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1908', 'Identity Theft via computer')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1909', 'Introduction of virus or contaminants into computer system/program')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1910', 'Minor solicited for sex via internet/Known minor')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1911', 'Theft of computer data')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1912', 'Threatening E-mail/Text Messages')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1913', 'Suspect meets victim on internet/chatroom')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1914', 'Unauthorized access to computer system')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1915', 'Internet Extortion')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('1916', 'Victim paid by wire transfer')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2000', 'Domestic violence')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2001', 'Suspect on drugs')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2002', 'Suspect intoxicated/drunk')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2003', 'Suspect 5150/mentally challenged or disturbed')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2004', 'Suspect is homeless/transient')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2005', 'Suspect uses wheelchair')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2006', 'Suspect was transgender')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2007', 'Suspect was homosexual/gay')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2008', 'In possession of a Ballistic vest')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2009', 'Suspect was Inmate/Incarcerated')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2010', 'Suspect was Jailer/Police Officer')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2011', 'Vendor (street or sidewalk)')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2012', 'costumed character (e.g., Barney, Darth Vader, Spiderman, etc.)')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2013', 'Tour Bus/Van Operator')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2014', 'Suspect was Uber/Lyft driver')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2015', 'Suspect was Foster child')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2016', 'Suspect was Train Operator')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2017', 'Suspect was MTA Bus Driver')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2018', 'Cannabis related')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2019', 'Theft of animal (non-livestock)')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2020', 'Mistreatment of animal')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2021', 'Suspect was Aged (60+over)')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2022', 'Suspect was Hitchhiker')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2023', 'Suspect was Prostitute')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2024', 'Suspect was Juvenile')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2025', 'Suspect was Bisexual')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2026', 'Suspect was Deaf/hearing impaired')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2027', 'Suspect was Pregnant')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2028', 'Suspect was Repeat/known shoplifter')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2029', 'Victim used profanity')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2030', 'Victim used racial slurs')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2031', 'Victim used hate-related language')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2032', 'Victim left property unattended')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2033', 'Victim refused to cooperate w/investigation')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2034', 'Victim was asleep/unconscious')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2035', 'Racial slurs')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2036', 'Hate-related language')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2037', 'Temporary/Vacation rental (AirBnB, etc)')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2038', 'Restraining order in place between suspect and victim')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2039', 'Victim was costumed character (e.g. Barney, Darth Vader, Spiderman, etc.)')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2040', 'Threats via Social Media')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2041', 'Harrassment via Social Media')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2042', 'Victim staying at short-term vacation rental')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2043', 'Victim is owner of short-term vacation rental')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2044', 'Suspect staying at short-term vacation rental')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2045', 'Suspect is owner of short-term vacation rental')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2046', 'Suspect damaged property equal to or exceeding $25,000')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2047', 'Victim was injured requiring transportation away from scene for medical reasons')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2048', 'Victim was on transit platform')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2049', 'Victim was passenger on bus')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2050', 'Victim was passenger on train')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2051', 'Suspect was passenger on bus')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('2052', 'Suspect was passenger on train')
    new_la_crime_incl_pred['mocodes'] = new_la_crime_incl_pred['mocodes'].str.replace('9999', 'Indistinctive MO')


    # Create new_la_crime_incl_modesc table
    from sqlalchemy import create_engine
    from sqlalchemy.types import Integer, Date
    engine = create_engine('sqlite:///db/la_crime.db')
    new_la_crime_incl_pred.to_sql('new_la_crime_w_desc', engine, if_exists='replace', dtype={"date_occ": Date(), "date_rptd": Date()})

    conn = sqlite3.connect("db/la_crime.db")
    c = conn.cursor()
    c.execute('''DROP TABLE new_la_crime_incl_modesc''')
    c.execute('''CREATE TABLE new_la_crime_incl_modesc
        ( dr_no BIGINT PRIMARY KEY,
        area_id BIGINT,
        area_name TEXT,
        crm_cd BIGINT,
        crm_cd_desc TEXT,
        cross_street TEXT,
        date_occ DATE,
        date_rptd DATE,
        location TEXT,
        longitude FLOAT,
        latitude FLOAT,
        mocodes TEXT,
        premis_cd BIGINT,
        premis_desc TEXT,
        rpt_dist_no BIGINT,
        status TEXT,
        status_desc TEXT,
        hour_occ BIGINT,
        minute_occ BIGINT,
        vict_age BIGINT,
        vict_descent TEXT,
        vict_sex TEXT,
        weapon_desc TEXT,
        weapon_used_cd FLOAT,
        loc_type TEXT,
        "FBI_Category" TEXT,
        "FBI_Cat_Prediction" TEXT ) ''')
    conn.commit()
    conn.close()

    conn = sqlite3.connect("db/la_crime.db")
    c = conn.cursor()
    c.execute('''INSERT INTO new_la_crime_incl_modesc SELECT * FROM new_la_crime_w_desc;''')
    conn.commit()
    conn.close()

    conn = sqlite3.connect("db/la_crime.db")
    c = conn.cursor()
    c.execute('''DROP TABLE new_la_crime_w_desc;''')
    conn.commit()
    conn.close()

    conn = sqlite3.connect("db/la_crime.db")
    c = conn.cursor()
    c.execute('''DROP TABLE IF EXISTS crime_accuracy_cnt;''')
    c.execute('''CREATE TABLE crime_accuracy_cnt(FBI_Category TEXT PRIMARY KEY, Accurate INT, Inaccurate INT, Total INT);''')
    c.execute('''INSERT INTO crime_accuracy_cnt select FBI_Category,
                 sum(case when fbi_cat_prediction = FBI_Category then 1 else 0 end) as Accurate,
                 sum(case when fbi_cat_prediction != FBI_Category then 1 else 0 end) as Inaccurate,
                 count(*)as Total
                 from new_la_crime_incl_modesc
                 group by FBI_Category;''')

    conn.commit()
    conn.close()


    # Redirect back to home page
    return redirect("/", code=302)


@app.route("/map")
def map():

    return render_template("map.html",xpage="map")

@app.route("/models")
def model():
    r"""Display the machine learning model results"""
    
    return render_template("models.html",xpage="Models")

@app.route("/data")
def data():
    r"""Display the data table"""
    
    return render_template("data.html",xpage="Data")

@app.route("/sources")
def sources():
    r"""Display the data sources"""
    
    return render_template("sources.html",xpage="Sources")

# @app.route("/mo_codes")
# def mo_codes():
#     r"""Display the mo codes"""
    
#     return render_template("mo_codes.html",xpage="MO Codes")
    
@app.route("/glossary")
def glossary():
    r"""Display the glossary"""
    
    return render_template("glossary.html",xpage="Glossary")

@app.route("/crime_stats/get_data")
def state_stats_get_data():

    r"""API backend that returns a json of the
    crime statistics for d3"""
    
    res = db.session.query(ZipCombinedStats).all()

    dlist = []

    for dset in res:
        md = dset.__dict__.copy()
        del md['_sa_instance_state']
        dlist.append(md)


    # find min and max for
    # selected columns
    min_max_list = ['household_median_income',
                    'pct_full_time_employed_pop',
                    'total_deaths_per_1000',
                    'opioid_rx_per_1000',
                    'adi_state_rank'
                    ]

    for item in min_max_list:

        dd = [ rec[item] for rec in dlist ]
        
        xmin = min(dd)
        xmax = max(dd)

        for i in range(len(dlist)):
            dlist[i][item+"_max"] = xmax
            dlist[i][item+"_min"] = xmin
        
    return jsonify(dlist)

@app.route("/crime_stats/results")
def crime_stats_results():

    r"""API backend that returns a json of the
    crime statistics for d3 stack graph"""
    
    res = db.session.query(CrimeAccuracy).all()

    dlist = []

    for dset in res:
        md = dset.__dict__.copy()
        del md['_sa_instance_state']
        dlist.append(md)
       
    return jsonify(dlist)


@app.route("/crime_sites")
def crime_sites():
    r""" This function returns the list of crime incidents
    with coordinates """
      
    res = db.session.query(LACrime).order_by('new_la_crime_incl_modesc.date_occ desc')

    dlist = []
    for dset in res:
        md = dset.__dict__.copy()
        del md['_sa_instance_state']
        dlist.append(md)
    
    return jsonify(dlist)

@app.route("/crime_sites/<dr_no>")
def sample_metadata(dr_no):
    """Return the metadata for a given crime dr_no."""
    sel = [
        LACrime.dr_no,
        LACrime.area_id,
        LACrime.area_name,
        LACrime.crm_cd,
        LACrime.crm_cd_desc,
        LACrime.cross_street,
        LACrime.date_occ,
        LACrime.date_rptd,
        LACrime.location,
        LACrime.longitude,
        LACrime.latitude,
        LACrime.mocodes,
        LACrime.premis_cd,
        LACrime.premis_desc,
        LACrime.rpt_dist_no,
        LACrime.status,
        LACrime.status_desc,
        LACrime.hour_occ,
        LACrime.vict_age,
        LACrime.vict_descent,
        LACrime.vict_sex,
        LACrime.weapon_desc,
        LACrime.weapon_used_cd,
        LACrime.FBI_Category,
        LACrime.FBI_Cat_Prediction
    ]

    results = db.session.query(*sel).filter(LACrime.dr_no == dr_no).all()

    # Create a dictionary entry for each row of metadata information
    sample_metadata = {}
    for result in results:
        sample_metadata["dr_no"] = result[0]
        sample_metadata["area_id"] = result[1]
        sample_metadata["area_name"] = result[2]
        sample_metadata["crm_cd"] = result[3]
        sample_metadata["crm_cd_desc"] = result[4]
        sample_metadata["cross_street"] = result[5]
        sample_metadata["date_occ"] = result[6]
        sample_metadata["date_rptd"] = result[7]
        sample_metadata["location"] = result[8]
        sample_metadata["longitude"] = result[9]
        sample_metadata["latitude"] = result[10]
        sample_metadata["mocodes"] = result[11]
        sample_metadata["premis_cd"] = result[12]
        sample_metadata["premis_desc"] = result[13]
        sample_metadata["rpt_dist_no"] = result[14]
        sample_metadata["status"] = result[15]
        sample_metadata["status_desc"] = result[16]
        sample_metadata["time_occ"] = result[17]
        sample_metadata["vict_age"] = result[18]
        sample_metadata["vict_descent"] = result[19]
        sample_metadata["vict_sex"] = result[20]
        sample_metadata["weapon_desc"] = result[21]
        sample_metadata["weapon_used_cd"] = result[22]
        sample_metadata["FBI_Category"] = result[23]
        sample_metadata["FBI_Cat_Prediction"] = result[24]
        
    print(sample_metadata)
    return jsonify(sample_metadata)

    

if __name__ == "__main__":
    app.run(debug=False)
