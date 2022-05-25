# Cities with Nice Weather v2.0

I recently came accross this blog article ["Cities with Nice Weather"](https://jdonland.github.io/city_temperatures/index.html) by ***Jesse Onland***.

He explains his approach on how to find the best cities for him to live in, in terms of climate.

What he's looking for is a city with an average temperature over a year close to that of Toronto, and a variance/range in temperature as small as possible.

His analysis is interesting, well written and organized, but the data he used is far from being exhaustive.

Indeed, the [list he got from Wikipedia](https://en.wikipedia.org/wiki/List_of_cities_by_average_temperature) only shows temperature data for major cities, so he might be missing his dream city without knowing it...

Moreover, the list only provides temperature data. There is no information about wind speed and relative humidity, which are two important factors influencing the perceived temperature.

Let's try the same approach with another dataset, and see if we can draw the same conclusions.

Some imports for later:


```python
%matplotlib inline
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycountry_convert as pc
from adjustText import adjust_text
from geopandas import GeoDataFrame
from meteostat import Normals, Point
from tqdm.notebook import tqdm
```


```python
Point.radius = 55_000  # Maximum radius for nearby stations in meters
plt.rcParams["figure.facecolor"] = (1.0, 1.0, 1.0, 1)
plt.rcParams["figure.figsize"] = [20, 10]
plt.rcParams["legend.fontsize"] = "x-large"
plt.rcParams["axes.titlesize"] = "x-large"
plt.rcParams["axes.labelsize"] = "x-large"
plt.rcParams["xtick.labelsize"] = "x-large"
plt.rcParams["ytick.labelsize"] = "x-large"
pd.set_option("display.max_rows", None)
```

## Get Data for Cities

First, we start with getting a list of cities with the name, country, population and coordinates since we're going to need latitude and longitude to retrieve historical weather data later on.

This kind of data is fortunately quite easy to find, and more importantly, free.

`dataset_1` is taken from [opendatasoft](https://public.opendatasoft.com/explore/dataset/geonames-all-cities-with-a-population-1000/table/?disjunctive.cou_name_en&sort=name). It contains information for 140,000 cities in the world.

`dataset_2`, from the [World Cities Database](https://simplemaps.com/data/world-cities), contains the same kind of data than `dataset_1`.


```python
dataset_1 = pd.read_csv(r"./data/geonames-all-cities-with-a-population-1000.csv", sep=";", na_filter=False)
dataset_2 = pd.read_csv(r"./data/simplemaps_worldcities_basicv1.75.csv", sep=",", na_filter=False)
# We need to set na_filter=False since pandas will convert country code "NA" for Namibia as NaN...
```


```python
dataset_1.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Geoname ID</th>
      <th>Name</th>
      <th>ASCII Name</th>
      <th>Alternate Names</th>
      <th>Feature Class</th>
      <th>Feature Code</th>
      <th>Country Code</th>
      <th>Country name EN</th>
      <th>Country Code 2</th>
      <th>Admin1 Code</th>
      <th>Admin2 Code</th>
      <th>Admin3 Code</th>
      <th>Admin4 Code</th>
      <th>population</th>
      <th>Elevation</th>
      <th>DIgital Elevation Model</th>
      <th>Timezone</th>
      <th>Modification date</th>
      <th>LABEL EN</th>
      <th>Coordinates</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8396129</td>
      <td>Sanjiang</td>
      <td>Sanjiang</td>
      <td>Sanjiang,Sanjiang Jiedao,Sanjiang Qu,san jiang...</td>
      <td>P</td>
      <td>PPLA3</td>
      <td>CN</td>
      <td>China</td>
      <td></td>
      <td>01</td>
      <td>3402</td>
      <td></td>
      <td></td>
      <td>0</td>
      <td></td>
      <td>14</td>
      <td>Asia/Shanghai</td>
      <td>2021-09-19</td>
      <td>China</td>
      <td>31.34813,118.36132</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8405692</td>
      <td>Xinmin</td>
      <td>Xinmin</td>
      <td>Xinmin,Xinmin Zhen,xin min,xin min zhen,新民,新民镇</td>
      <td>P</td>
      <td>PPLA4</td>
      <td>CN</td>
      <td>China</td>
      <td></td>
      <td>33</td>
      <td>8739734</td>
      <td></td>
      <td></td>
      <td>28033</td>
      <td></td>
      <td>402</td>
      <td>Asia/Shanghai</td>
      <td>2022-04-12</td>
      <td>China</td>
      <td>30.39759,107.3895</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8416824</td>
      <td>Jindaoxia</td>
      <td>Jindaoxia</td>
      <td>Jindaoxia,Jindaoxia Zhen,jin dao xia,jin dao x...</td>
      <td>P</td>
      <td>PPLA4</td>
      <td>CN</td>
      <td>China</td>
      <td></td>
      <td>33</td>
      <td>8739734</td>
      <td></td>
      <td></td>
      <td>13752</td>
      <td></td>
      <td>323</td>
      <td>Asia/Shanghai</td>
      <td>2022-04-01</td>
      <td>China</td>
      <td>30.00528,106.65187</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8420197</td>
      <td>Jianlong</td>
      <td>Jianlong</td>
      <td>Jianlong,Jianlong Xiang,jian long,jian long xi...</td>
      <td>P</td>
      <td>PPLA4</td>
      <td>CN</td>
      <td>China</td>
      <td></td>
      <td>33</td>
      <td>8739734</td>
      <td></td>
      <td></td>
      <td>18151</td>
      <td></td>
      <td>276</td>
      <td>Asia/Shanghai</td>
      <td>2022-04-01</td>
      <td>China</td>
      <td>29.3586,106.18522</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8505210</td>
      <td>Jianhua</td>
      <td>Jianhua</td>
      <td>Bukui,Bukui Jiedao,Jianhua,Jianhua Qu,bo kui,b...</td>
      <td>P</td>
      <td>PPLA3</td>
      <td>CN</td>
      <td>China</td>
      <td></td>
      <td>08</td>
      <td>2302</td>
      <td></td>
      <td></td>
      <td>0</td>
      <td></td>
      <td>146</td>
      <td>Asia/Shanghai</td>
      <td>2022-03-12</td>
      <td>China</td>
      <td>47.35773,123.95977</td>
    </tr>
  </tbody>
</table>
</div>



Would be nice to split the `Coordinates` column:


```python
dataset_1["Coordinates"] = dataset_1["Coordinates"].astype("string")
dataset_1[["lat", "lng"]] = dataset_1["Coordinates"].str.split(pat=",", n=1, expand=True)
dataset_1["lat"] = dataset_1["lat"].astype(float)
dataset_1["lng"] = dataset_1["lng"].astype(float)
```

We drop useless data:


```python
dataset_1.drop(
    columns=[
        "Geoname ID",
        "Name",
        "Alternate Names",
        "Feature Class",
        "Feature Code",
        "Country Code 2",
        "Admin1 Code",
        "Admin2 Code",
        "Admin3 Code",
        "Admin4 Code",
        "Elevation",
        "DIgital Elevation Model",
        "Timezone",
        "LABEL EN",
        "Modification date",
        "Coordinates",
    ],
    inplace=True,
)
```

Some entries don't have a country name, we remove them:


```python
dataset_1.query("`Country name EN` != ''", inplace=True)
```

Done with `dataset_1`! Let's clean the second one:


```python
dataset_2.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>city_ascii</th>
      <th>lat</th>
      <th>lng</th>
      <th>country</th>
      <th>iso2</th>
      <th>iso3</th>
      <th>admin_name</th>
      <th>capital</th>
      <th>population</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Tokyo</td>
      <td>Tokyo</td>
      <td>35.6839</td>
      <td>139.7744</td>
      <td>Japan</td>
      <td>JP</td>
      <td>JPN</td>
      <td>Tōkyō</td>
      <td>primary</td>
      <td>39105000</td>
      <td>1392685764</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jakarta</td>
      <td>Jakarta</td>
      <td>-6.2146</td>
      <td>106.8451</td>
      <td>Indonesia</td>
      <td>ID</td>
      <td>IDN</td>
      <td>Jakarta</td>
      <td>primary</td>
      <td>35362000</td>
      <td>1360771077</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Delhi</td>
      <td>Delhi</td>
      <td>28.6667</td>
      <td>77.2167</td>
      <td>India</td>
      <td>IN</td>
      <td>IND</td>
      <td>Delhi</td>
      <td>admin</td>
      <td>31870000</td>
      <td>1356872604</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Manila</td>
      <td>Manila</td>
      <td>14.6000</td>
      <td>120.9833</td>
      <td>Philippines</td>
      <td>PH</td>
      <td>PHL</td>
      <td>Manila</td>
      <td>primary</td>
      <td>23971000</td>
      <td>1608618140</td>
    </tr>
    <tr>
      <th>4</th>
      <td>São Paulo</td>
      <td>Sao Paulo</td>
      <td>-23.5504</td>
      <td>-46.6339</td>
      <td>Brazil</td>
      <td>BR</td>
      <td>BRA</td>
      <td>São Paulo</td>
      <td>admin</td>
      <td>22495000</td>
      <td>1076532519</td>
    </tr>
  </tbody>
</table>
</div>



Again, we drop useless columns:


```python
dataset_2.drop(columns=["city", "iso2", "iso3", "admin_name", "capital", "id"], inplace=True)
```

Some cities lack population data, we remove them:


```python
dataset_2 = dataset_2[dataset_2["population"].str.isnumeric()]
dataset_2["population"] = dataset_2["population"].astype(int)
```

Looks like `dataset_1` is more exhaustive than `dataset_2`:


```python
print(dataset_1.shape, dataset_2.shape)
```

    (139571, 6) (42143, 5)
    

Let's check the data for a small city in UK:


```python
dataset_1.query("`Country Code` == 'GB' and `ASCII Name` == 'Portsmouth'")
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ASCII Name</th>
      <th>Country Code</th>
      <th>Country name EN</th>
      <th>population</th>
      <th>lat</th>
      <th>lng</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>66901</th>
      <td>Portsmouth</td>
      <td>GB</td>
      <td>United Kingdom</td>
      <td>194150</td>
      <td>50.79899</td>
      <td>-1.09125</td>
    </tr>
  </tbody>
</table>
</div>




```python
dataset_2.query("`country` == 'United Kingdom' and city_ascii == 'Portsmouth'")
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city_ascii</th>
      <th>lat</th>
      <th>lng</th>
      <th>country</th>
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2407</th>
      <td>Portsmouth</td>
      <td>50.8058</td>
      <td>-1.0872</td>
      <td>United Kingdom</td>
      <td>248440</td>
    </tr>
  </tbody>
</table>
</div>



Not the same figures but seems pretty close to reality.

What about Tokyo:


```python
dataset_1.query("`Country Code` == 'JP' and `ASCII Name` == 'Tokyo'")
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ASCII Name</th>
      <th>Country Code</th>
      <th>Country name EN</th>
      <th>population</th>
      <th>lat</th>
      <th>lng</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>33145</th>
      <td>Tokyo</td>
      <td>JP</td>
      <td>Japan</td>
      <td>8336599</td>
      <td>35.6895</td>
      <td>139.69171</td>
    </tr>
  </tbody>
</table>
</div>




```python
dataset_2.query("`country` == 'Japan' and city_ascii == 'Tokyo'")
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city_ascii</th>
      <th>lat</th>
      <th>lng</th>
      <th>country</th>
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Tokyo</td>
      <td>35.6839</td>
      <td>139.7744</td>
      <td>Japan</td>
      <td>39105000</td>
    </tr>
  </tbody>
</table>
</div>



This time, the difference is not negligible, there's a huge difference in terms of population size.

Looks like the `dataset_1` considers the 23 wards that made up the boundaries of the historic city of Tokyo, while `dataset_2` considers the greater Tokyo metropolitan area, which is spread over 3 prefectures...

We should notice the same issue with Manila, capital city of the Philippines:


```python
dataset_1.query("`Country Code` == 'PH' and `ASCII Name` == 'Manila'")
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ASCII Name</th>
      <th>Country Code</th>
      <th>Country name EN</th>
      <th>population</th>
      <th>lat</th>
      <th>lng</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16765</th>
      <td>Manila</td>
      <td>PH</td>
      <td>Philippines</td>
      <td>1600000</td>
      <td>14.6042</td>
      <td>120.9822</td>
    </tr>
  </tbody>
</table>
</div>




```python
dataset_2.query("`country` == 'Philippines' and city_ascii == 'Manila'")
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city_ascii</th>
      <th>lat</th>
      <th>lng</th>
      <th>country</th>
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>Manila</td>
      <td>14.6</td>
      <td>120.9833</td>
      <td>Philippines</td>
      <td>23971000</td>
    </tr>
  </tbody>
</table>
</div>



Yes, same issue: `dataset_1` considers the Manila city, while `dataset_2` gives data for the larger urban area of Metro Manila.

This explains why `dataset_2` was smaller, it merges cities of large urban areas.

Last check with Singapore:


```python
dataset_1.query("`Country Code` == 'SG' and `ASCII Name` == 'Singapore'")
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ASCII Name</th>
      <th>Country Code</th>
      <th>Country name EN</th>
      <th>population</th>
      <th>lat</th>
      <th>lng</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9440</th>
      <td>Singapore</td>
      <td>SG</td>
      <td>Singapore</td>
      <td>3547809</td>
      <td>1.28967</td>
      <td>103.85007</td>
    </tr>
  </tbody>
</table>
</div>




```python
dataset_2.query("`country` == 'Singapore'")
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city_ascii</th>
      <th>lat</th>
      <th>lng</th>
      <th>country</th>
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>135</th>
      <td>Singapore</td>
      <td>1.3</td>
      <td>103.8</td>
      <td>Singapore</td>
      <td>5271000</td>
    </tr>
  </tbody>
</table>
</div>



Both datasets seem outdated since the current population of Singapore is 6M, for both the country and the city.

Some data visualization would help us see if some cities/countries/continents are missing:


```python
plt.scatter(x=dataset_1["lng"], y=dataset_1["lat"], s=dataset_1["population"] / 1e6)
plt.show()
```


    
![png](README_files/README_41_0.png)
    


People sure like living near the sea!

Same plot with `dataset_2` this time:


```python
plt.scatter(x=dataset_2["lng"], y=dataset_2["lat"], s=dataset_2["population"] / 1e6)
plt.show()
```


    
![png](README_files/README_44_0.png)
    


Since both datasets seem accurate enough, from now on we'll just keep the first dataset.


```python
df = dataset_1.copy()
del dataset_1
del dataset_2
print(df.shape)
df.head()
```

    (139571, 6)
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ASCII Name</th>
      <th>Country Code</th>
      <th>Country name EN</th>
      <th>population</th>
      <th>lat</th>
      <th>lng</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sanjiang</td>
      <td>CN</td>
      <td>China</td>
      <td>0</td>
      <td>31.34813</td>
      <td>118.36132</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Xinmin</td>
      <td>CN</td>
      <td>China</td>
      <td>28033</td>
      <td>30.39759</td>
      <td>107.38950</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jindaoxia</td>
      <td>CN</td>
      <td>China</td>
      <td>13752</td>
      <td>30.00528</td>
      <td>106.65187</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jianlong</td>
      <td>CN</td>
      <td>China</td>
      <td>18151</td>
      <td>29.35860</td>
      <td>106.18522</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Jianhua</td>
      <td>CN</td>
      <td>China</td>
      <td>0</td>
      <td>47.35773</td>
      <td>123.95977</td>
    </tr>
  </tbody>
</table>
</div>



We add a column for the continent code, which might be useful later:


```python
df["Country Code"] = df["Country Code"].astype("string")
df.query("`Country Code` != 'TL'", inplace=True)  # Remove Timor-Leste, not supported by pycountry
df.query("`Country Code` != 'EH'", inplace=True)  # Remove Western Sahara, not supported by pycountry
df["Continent Code"] = df.apply(lambda row: pc.country_alpha2_to_continent_code(row["Country Code"]), axis=1)
```

Number of cities per continent:


```python
print(df.shape)
df["Continent Code"].value_counts()
```

    (139552, 7)
    




    EU    69029
    NA    29755
    AS    24765
    SA     6701
    AF     4889
    OC     4413
    Name: Continent Code, dtype: int64



## Get Historial Weather Data

We now need some historical weather data.

As explained earlier, we can't use [this list](https://en.wikipedia.org/wiki/List_of_cities_by_average_temperature) since it only contains data for "big" cities.

We need historical weather data of the last decade for all the cities shortlisted.

Only monthly statistics are needed since temperatures don't change significantly within a month. We'll compute the variance in temperature of a city only from the statistical monthly weather data.

[OpenWeather](https://openweathermap.org/) provides an API with [statistical monthly weather data](https://openweathermap.org/api/statistics-api#month) (temp, humidity, wind, etc.) for any month of the entire year, but it's way too expensive for us.

For now, we'll use [Meteostat](https://dev.meteostat.net/), an open platform which provides free access to historical weather and climate data. Unfortunately, it lacks information about wind speed or humidity.

Getting weather data for Tokyo is straightforward:


```python
tokyo = Point(lat=35.6839, lon=139.7744)
data = Normals(tokyo, 1991, 2020).fetch()
data
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tavg</th>
      <th>tmin</th>
      <th>tmax</th>
      <th>prcp</th>
      <th>wspd</th>
      <th>pres</th>
      <th>tsun</th>
    </tr>
    <tr>
      <th>month</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>6.3</td>
      <td>2.6</td>
      <td>10.0</td>
      <td>59.8</td>
      <td>NaN</td>
      <td>1015.9</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.0</td>
      <td>3.1</td>
      <td>10.8</td>
      <td>56.6</td>
      <td>NaN</td>
      <td>1015.9</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.0</td>
      <td>5.9</td>
      <td>14.0</td>
      <td>116.6</td>
      <td>NaN</td>
      <td>1015.1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14.8</td>
      <td>10.7</td>
      <td>19.0</td>
      <td>134.1</td>
      <td>NaN</td>
      <td>1013.9</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>19.6</td>
      <td>15.7</td>
      <td>23.5</td>
      <td>139.9</td>
      <td>NaN</td>
      <td>1011.9</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>22.8</td>
      <td>19.5</td>
      <td>26.0</td>
      <td>168.0</td>
      <td>NaN</td>
      <td>1009.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>26.7</td>
      <td>23.4</td>
      <td>30.0</td>
      <td>156.5</td>
      <td>NaN</td>
      <td>1008.6</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>28.1</td>
      <td>24.7</td>
      <td>31.5</td>
      <td>154.7</td>
      <td>NaN</td>
      <td>1010.2</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>24.4</td>
      <td>21.2</td>
      <td>27.6</td>
      <td>222.4</td>
      <td>NaN</td>
      <td>1013.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>19.0</td>
      <td>15.8</td>
      <td>22.2</td>
      <td>231.3</td>
      <td>NaN</td>
      <td>1016.6</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>13.8</td>
      <td>10.3</td>
      <td>17.2</td>
      <td>96.5</td>
      <td>NaN</td>
      <td>1018.1</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>8.7</td>
      <td>5.0</td>
      <td>12.4</td>
      <td>58.1</td>
      <td>NaN</td>
      <td>1016.9</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.plot(y=["tavg", "tmin", "tmax"])
plt.ylabel("Temperature °C")
plt.show()
```


    
![png](README_files/README_56_0.png)
    


Pretty big variations over the year for Tokyo!

Having lived there for a few weeks, the `tmax` data is closer to ~~the~~ my perceived temperature than the `tavg`.

We code a simple function to get weather data for a specific location:


```python
def get_mean_std_temperature_for_loc(lat: float, lon: float):
    pt = Point(lat=lat, lon=lon)
    try:
        with warnings.catch_warnings():  # disable some meteostat warnings
            warnings.simplefilter("ignore")
            data = Normals(pt, 1991, 2020).fetch()
    except pd.errors.ParserError:  # escape some meteostat bugs
        return np.nan, np.nan, np.nan, np.nan
    mean_ = data["tmax"].mean()
    std_ = data["tmax"].std()
    tmax_ = data["tmax"].max()
    range_ = data["tmax"].max() - data["tmax"].min()
    return mean_, std_, tmax_, range_
```

Gathering weather data for 140,000 cities takes some time. We keep the biggest cities:


```python
df.query("population >= 100000", inplace=True)
```

We iterate over cities in the dataframe and get weather data. Not the most efficient way to do it but it works...:


```python
Tmean = []
Tstd = []
Tmax = []
Trange = []
for row in tqdm(df.itertuples(), total=df.shape[0]):
    mean_, std_, tmax_, range_ = get_mean_std_temperature_for_loc(row.lat, row.lng)
    Tmean.append(mean_)
    Tstd.append(std_)
    Tmax.append(tmax_)
    Trange.append(range_)
df["Tmean"] = Tmean
df["Tstd"] = Tstd
df["Tmax"] = Tmax
df["Trange"] = Trange
```




Some cities don't have any weather data at all, we drop them:


```python
df.dropna(subset=["Tmean", "Tstd", "Tmax"], inplace=True)
```

## Summmary Statistics


```python
fig, axes = plt.subplots(2, 2)

df[["Tmean"]].plot.kde(ax=axes[0, 0])
axes[0, 0].set_xlabel("Mean temperature over a year")
axes[0, 0].get_legend().remove()

df[["Tmax"]].plot.kde(ax=axes[0, 1])
axes[0, 1].set_xlabel("Max temperature over a year")
axes[0, 1].get_legend().remove()

df[["Tstd"]].plot.kde(ax=axes[1, 1])
axes[1, 1].set_xlabel("Std dev in temperature over a year")
axes[1, 1].get_legend().remove()

df[["Trange"]].plot.kde(ax=axes[1, 0])
axes[1, 0].set_xlabel("Range in temperature over a year")
axes[1, 0].get_legend().remove()

plt.show()
```


    
![png](README_files/README_67_0.png)
    


Out of curiosity, let's get the extreme values in temperature:


```python
df.query("Tmean == Tmean.max() or Tmean == Tmean.min()")[["ASCII Name", "Country name EN", "Tmean", "Tstd"]]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ASCII Name</th>
      <th>Country name EN</th>
      <th>Tmean</th>
      <th>Tstd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>37323</th>
      <td>Mecca</td>
      <td>Saudi Arabia</td>
      <td>38.683333</td>
      <td>4.705864</td>
    </tr>
    <tr>
      <th>49692</th>
      <td>Yakutsk</td>
      <td>Russian Federation</td>
      <td>-2.883333</td>
      <td>22.694887</td>
    </tr>
  </tbody>
</table>
</div>



Cities with the most stable/unstable temperatures over a year:


```python
df.query("Tstd == Tstd.min() or Tstd == Tstd.max()")[["ASCII Name", "Country name EN", "Tmean", "Tstd"]]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ASCII Name</th>
      <th>Country name EN</th>
      <th>Tmean</th>
      <th>Tstd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>49692</th>
      <td>Yakutsk</td>
      <td>Russian Federation</td>
      <td>-2.883333</td>
      <td>22.694887</td>
    </tr>
    <tr>
      <th>122628</th>
      <td>Jayapura</td>
      <td>Indonesia</td>
      <td>32.241667</td>
      <td>0.317543</td>
    </tr>
  </tbody>
</table>
</div>



Yakutsk seems a like lovely city to live in...

Get some cities to plot later:


```python
cities_to_plot = [
    df.query("`ASCII Name` == 'Reykjavik' and `Country name EN` == 'Iceland'"),
    df.query("`ASCII Name` == 'Yakutsk' and `Country name EN` == 'Russian Federation'"),
    df.query("`ASCII Name` == 'Edinburgh' and `Country name EN` == 'United Kingdom'"),
    df.query("`ASCII Name` == 'Paris' and `Country name EN` == 'France'"),
    df.query("`ASCII Name` == 'Oslo' and `Country name EN` == 'Norway'"),
    df.query("`ASCII Name` == 'Stockholm' and `Country name EN` == 'Sweden'"),
    df.query("`ASCII Name` == 'Helsinki' and `Country name EN` == 'Finland'"),
    df.query("`ASCII Name` == 'Tokyo' and `Country name EN` == 'Japan'"),
    df.query("`ASCII Name` == 'Toronto' and `Country name EN` == 'Canada'"),
    df.query("`ASCII Name` == 'Singapore' and `Country name EN` == 'Singapore'"),
]
assert all(not city.empty for city in cities_to_plot)
```


```python
plt.scatter(df["Tmean"], df["Tstd"], s=5, c="k", marker=".")
plt.xlabel("Mean temperature")
plt.ylabel("Temperature standard deviation")
for city in cities_to_plot:
    plt.scatter(
        city["Tmean"].iloc[0],
        city["Tstd"].iloc[0],
        s=300,
        label=f'{city["ASCII Name"].iloc[0]} ({city["Country Code"].iloc[0]})',
    )
plt.legend()
plt.show()
```


    
![png](README_files/README_75_0.png)
    


Nice inverse correlation!

We plot the same data on a map:


```python
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
ax = world.boundary.plot(edgecolor="black")
sc = plt.scatter(x=df["lng"], y=df["lat"], c=df["Tmean"], s=100, cmap="winter")
plt.colorbar(sc, label="Mean temperature °C")
plt.show()
```


    
![png](README_files/README_78_0.png)
    


Seems like many cities in Africa and South America disappeared for lack of data.

## So Which Cities Are the "Best"?

*Disclaimer: everything below is based only on my personal preferences...*

Set some threshold to remove cities with warm weather:


```python
df_backup = df.copy()
df.query("Tmean <= 17 and Tmax <= 30 and Tstd <= 10 and Trange <= 14", inplace=True)
```

Most shortlisted cities are in Europe:


```python
print(df.shape)
print(df.groupby(["Continent Code"])["Country name EN"].value_counts())
```

    (37, 11)
    Continent Code  Country name EN
    EU              United Kingdom     28
                    Ireland             2
                    France              1
                    Iceland             1
    NA              Canada              3
    OC              New Zealand         2
    Name: Country name EN, dtype: int64
    

Canada shortlisted cities were under the [2021 heat wave](https://en.wikipedia.org/wiki/2021_Western_North_America_heat_wave). No thanks, I'll pass.


```python
df_europe = df.query("`Continent Code` == 'EU'")
df_nz = df.query("`Continent Code` == 'OC'")
```


```python
fig, (ax1, ax2) = plt.subplots(1, 2)
world.boundary.plot(ax=ax1, edgecolor="black")
world.boundary.plot(ax=ax2, edgecolor="black")
sc1 = ax1.scatter(x=df_europe["lng"], y=df_europe["lat"], c=df_europe["Tmean"], s=100, cmap="winter")
sc2 = ax2.scatter(x=df_nz["lng"], y=df_nz["lat"], c=df_nz["Tmean"], s=100, cmap="winter")
ax1.set_xlim(-25, 5)
ax1.set_ylim(40, 70)
ax2.set_xlim(160, 180)
ax2.set_ylim(-50, -30)
cbar = fig.colorbar(sc2, ax=[ax1, ax2])
cbar.set_label("Mean temperature °C")
ax1.set_title("Europe")
ax2.set_title("New Zealand")
plt.show()
```


    
![png](README_files/README_88_0.png)
    


I was expecting to see Ireland and Scotland so I'm not surprised.

I however thought some cities from Norway/Sweden/Finland would have been shortlisted. Looks like they have a higher temperature range:


```python
df_shortlisted = pd.concat((df_europe, df_nz))
df_nordic = df_backup.query(
    "`Country name EN` == 'Norway' or `Country name EN` == 'Sweden' or `Country name EN` == 'Finland'"
)

fig, axes = plt.subplots(2, 2)

df_nordic[["Tmean"]].plot.kde(ax=axes[0, 0])
df_shortlisted[["Tmean"]].plot.kde(ax=axes[0, 0])
axes[0, 0].legend(["Nordic (NO/SE/FI)", "IS/IE/UK/FR/NZ"])
axes[0, 0].set_xlabel("Mean temperature over a year")

df_nordic[["Tmax"]].plot.kde(ax=axes[0, 1])
df_shortlisted[["Tmax"]].plot.kde(ax=axes[0, 1])
axes[0, 1].legend(["Nordic (NO/SE/FI)", "IS/IE/UK/FR/NZ"])
axes[0, 1].set_xlabel("Max temperature over a year")

df_nordic[["Tstd"]].plot.kde(ax=axes[1, 1])
df_shortlisted[["Tstd"]].plot.kde(ax=axes[1, 1])
axes[1, 1].legend(["Nordic (NO/SE/FI)", "IS/IE/UK/FR/NZ"])
axes[1, 1].set_xlabel("Std dev in temperature over a year")

df_nordic[["Trange"]].plot.kde(ax=axes[1, 0])
df_shortlisted[["Trange"]].plot.kde(ax=axes[1, 0])
axes[1, 0].legend(["Nordic (NO/SE/FI)", "IS/IE/UK/FR/NZ"])
axes[1, 0].set_xlabel("Range in temperature over a year")

plt.show()
```


    
![png](README_files/README_90_0.png)
    


UK/IS/IE seem to have the "best" weather for me.

Possible future steps: Include pressure/wind to have the perceived temperature or heat index.
