import requests
import csv
import re
import time

def get_movie_title():
	with open('movies_2016.txt') as f:
		for line in f:
			yield line

movie_generator = get_movie_title()
def write_to_csv_file():
	""" It iterates through the list of movies from a file and makes the request to API and saves the name of 
	    director, actors, ratings which in turn gets saved in a .csv file """
	for movie in movie_generator:
		resp = requests.get('https://www.omdbapi.com/?t={}'.format(movie))
		resp_to_json = resp.json()

		
		if resp_to_json['Response'] == 'False':
			write_errors(movie)
			continue
		try:
			runtime = resp_to_json['Runtime'].strip(' min')
		except ValueError:
			runtime = 'N/A'

		director = resp_to_json['Director']
		writer = re.sub(r'\([\w\s]*\)', '', resp_to_json['Writer'])
		actors = resp_to_json['Actors']
		title = resp_to_json['Title']
		try:
			imdb_rating = float(resp_to_json['imdbRating'])
		except ValueError:
			imdb_rating = 'N/A'

		if any(value for value in [title, runtime, director, writer, actors, imdb_rating] if value == 'N/A'):
			write_errors(movie)
			continue


		if imdb_rating >= 6.8:
			target = 1 #hit
		else:
			target = 0 #flop


		with open('movie_2016.csv', 'a') as f:
			wr = csv.writer(f, delimiter=',')
			wr.writerow((title, runtime, director, writer, actors, target))

		

def write_errors(title):
	with open('movies_2016_errors.txt', 'a') as f:
		f.write(title)

write_to_csv_file()