from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int

new_person = { 'name': 'nitish', 'age': 35}
new_person2 = { 'name': 'nitish', 'age': '35'}

print(new_person, new_person2)

# even after specifying to be int, str also works
