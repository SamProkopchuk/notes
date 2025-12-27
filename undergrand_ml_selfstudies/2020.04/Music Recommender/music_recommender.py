import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import cosine_similarity

"""
This program's occasional odd suggestions
are presumably due to a lack of data.

Also, for some reason, it recommends death metal and the like
when a user enters an intrumental artist, ex: Pretty Lights.
"""

artists_users_filename = "artist_users.npy"
art_ID_row_filename = "art_ID_row.pickle"
K = 3


def get_preprocessed_data(
        user_artists_df: pd.DataFrame,
        artists_df: pd.DataFrame):
    """
    Loads the preprocessed data if it's already been saved.
    Otherwise create and save the preprocessed data.

    Returns
    - Processed user_artists_df matrix such that:
     - each row is an artist.
     - each column represents a user.
     - every cell represents number of listens.

    - Dictionary of artist IDs to row of artist in the 
        artists_users matrix. 
    """
    # First check if saved matrix and dict exist
    try:
        artists_users = np.load(artists_users_filename)
        with open(art_ID_row_filename, "rb") as file:
            art_ID_row = pickle.load(file)
    except FileNotFoundError:
        print("Preprocessed data files not found.")
        print("Preprocessing data...")
        unique_users = user_artists_df["userID"].nunique()
        unique_artists = user_artists_df["artistID"].nunique()
        # Using this *BAD* implementation until I find a better alternative:
        artists_users = np.zeros((unique_artists, unique_users))
        artist_row, user_row = -1, -1
        art_ID_row, last_userID = {}, -1
        for _, row in user_artists_df.iterrows():
            if row["userID"] != last_userID:
                user_row += 1
                last_userID = row["userID"]
            if row["artistID"] not in art_ID_row:
                artist_row += 1
                art_ID_row[row["artistID"]] = artist_row
            artists_users[artist_row][user_row] = 1
        print("Done.")
        if input(
                ("Would you like to save the preprocessed data?: ") +
                ("Please note it will take between 250 and 300 MB of storage. ") +
                ("[Y/n]")) in ("y", "Y", ""):
            print("Saving preprocessed data...")
            np.save(artists_users_filename, artists_users)
            with open(art_ID_row_filename, "wb") as file:
                pickle.dump(art_ID_row, file, protocol=pickle.HIGHEST_PROTOCOL)
            print("Done.")
    return artists_users, art_ID_row


def main():
    try:
        user_artists_df = pd.read_table(
            "hetrec2011-lastfm-2k/user_artists.dat")
        artists_df = pd.read_table("hetrec2011-lastfm-2k/artists.dat")
    except FileNotFoundError:
        print("The required datasets were not found!")
        exit()
    artists_users, art_ID_row = get_preprocessed_data(
        user_artists_df, artists_df)
    art_row_ID = dict(map(lambda x: x[::-1], art_ID_row.items()))
    artists_users_scaled = scale(
        artists_users, axis=1, with_mean=True, with_std=True, copy=True)
    artists_users_scaled = np.where(
        artists_users == 0, 0, artists_users_scaled)
    print("Please NOTE: The dataset used isn't huge so it's recommended to use more popular artists.")
    while True:
        search_artist = input(
            "Name a music artist that you'd like to find similar ones to: ")
        if search_artist in artists_df["name"].values:
            print("Artist succesfully added to search.")
            break
        print("Artist not found in DB (Check spelling and Capitalization).")
    artist_df_row = np.where(artists_df["name"] == search_artist)[0]
    artist_id = artists_df["id"].values[artist_df_row][0]
    artist_row = art_ID_row[artist_id]
    similarities = cosine_similarity(
        artists_users_scaled[artist_row].reshape(1, -1), artists_users_scaled)[0]

    k_similar = np.argpartition(similarities, -K)[-K:]
    for sim in k_similar:
        print(similarities[sim])
    for sim in k_similar:
        artist = artists_df.loc[artists_df["id"] == art_row_ID[sim], "name"]
        print(artist)


if __name__ == "__main__":
    main()
