import requests
import time

""" This script scrapes all the titles from given year with required number of pages and saves them in a file"""
year = ['{:02d}'.format(x) for x in range(5,16)]
#year = [16]
#page = range(1,41)
page=[16]
#page = range(1,3)
with open('movies_2016.txt', 'a') as f:
	for yr in year:
		for pg in page:
			url = 'https://api.themoviedb.org/3/discover/movie?api_key=77c12055e873a3470f8b104fd41f7281&language=en-US&sort_by=popularity.desc&include_adult=false&include_video=false&page={}&vote_average.gte=5.0&year=20{}&with_runtime.gte=80'.format(pg,yr)
			#print (url)
			resp = requests.get(url)
			if str(resp.status_code).startswith('4'):
				with open('2016.txt', 'a') as f:
					f.write('Year;{} Page{},'.format(yr,pg))
					continue
			resp_json = resp.json()
			for movie in resp_json['results']:
				f.write(movie['title']+'\n')
			
		print ('Scrapped data from year ', yr)
		