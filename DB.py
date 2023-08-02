from pymongo import MongoClient

# Connect to the DB
client = MongoClient('localhost', 27017)
db = client.waste
todos = db.todos
tasks = db.tasks


def insert_to_db(collection, key, value) -> None:
    collection.insert_one({key: value})


def get_all_collection(collection) -> list:
    return list(collection.find({}))


def get_by_query(collection, query: dict) -> list:
    return list(collection.find(query))


# insert_to_db(tasks, "active", False)
# print(get_by_query(tasks, {"active": False}))
