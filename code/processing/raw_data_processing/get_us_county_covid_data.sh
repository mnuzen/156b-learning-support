#!/bin/bash
cd $(git rev-parse --show-toplevel)
wget https://static.usafacts.org/public/data/covid-19/covid_confirmed_usafacts.csv -O data/us/confirmed_cases.csv
wget https://static.usafacts.org/public/data/covid-19/covid_deaths_usafacts.csv -O data/us/deaths.csv