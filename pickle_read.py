import pickle
data = None
# Open the pickle file for reading in binary mode
def ne():
    with open('parsed_data.pkl', 'rb') as file:
        # Load the object stored in the pickle file
        data = pickle.load(file)
        print(data)

# Now you can use the 'data' variable which contains the object loaded from the pickle file
if __name__ == "__main__":
    ne()
    # print(data)