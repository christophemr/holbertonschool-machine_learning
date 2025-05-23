-- Lists all genres in the database by their rating
-- uses hbtn_0d_tvshows_rate database
-- database name will be passed as an argument of the mysql command

SELECT tv_genres.name, SUM(tv_show_ratings.rate) AS rating
FROM tv_genres
LEFT JOIN tv_show_genres ON tv_genres.id = tv_show_genres.genre_id
LEFT JOIN tv_show_ratings ON tv_show_genres.show_id = tv_show_ratings.show_id
GROUP BY tv_genres.name
ORDER BY rating DESC;
