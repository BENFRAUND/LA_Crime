#!/usr/bin/env python3

import pandas as pd
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

app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///db/la_crime.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

Base = automap_base(metadata=db.metadata)
engine = db.get_engine()
Base.prepare(engine, reflect=True)
LACrime = Base.classes.new_la_crime
ZipCombinedStats = Base.classes.la_zip_stats
# LARealEstateStats = Base.classes.la_real_estate


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/new_la_crime_data")
def refresh_data():
    # Collect refreshed data from LA crime API (https://data.lacity.org/resource/7fvc-faax.json?)
    incidents = requests.get("https://data.lacity.org/resource/7fvc-faax.json?$$app_token=" + lacrime_api_key + "&$order=date_occ%20DESC").json()
    
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
    
    # Redirect back to home page
    return redirect("/", code=302)


@app.route("/map")
def map():

    return render_template("map.html",xpage="map")

@app.route("/crime_stats")
def state_stats():
    r"""Display the crime stats plot"""
    
    return render_template("crime_stats.html",xpage="crime stats")

@app.route("/data")
def data():
    r"""Display the data table plot"""
    
    return render_template("data.html",xpage="Data")

@app.route("/data_sources")
def data_sources():
    r"""Display the data sources"""
    
    return render_template("data_sources.html",xpage="Data Sources")


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



@app.route("/crime_sites")
def crime_sites():
    r""" This function returns the list of crime incidents
    with coordinates """
      
    res = db.session.query(LACrime).order_by('new_la_crime.date_occ desc')

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
        LACrime.weapon_used_cd
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
        
    print(sample_metadata)
    return jsonify(sample_metadata)

    

if __name__ == "__main__":
    app.run(debug=True)
