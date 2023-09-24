""" Denne python fil er en eksempel løsning på opgaverne fra
Exercise 2. Der kan være andre varianter at løse opgaverne på, men dette
er mit bud"""

forfattere = ["Asimov", "Dostojevski", "Philip k. Dick", "Pushkin", "Tolkien"]
for forfatter in forfattere:
    print(forfatter)


forfattere.append("H.C. Andersen")
for forfatter in forfattere:
    print(forfatter)
del forfattere[1]
for forfatter in forfattere:
    print(forfatter)

antal = len(forfattere)
print("antallet af forfattere er : "+str(antal))


forfattere.reverse()
for forfatter in forfattere:
    print(forfatter)

tal_liste = range(1,11)
for tal in tal_liste:
    print(tal)
nyliste = range(3,100,3)
for tal in nyliste:
    print(tal)