""" Denne python fil er en eksempel l�sning p� opgaverne fra
Exercise 2. Der kan v�re andre varianter at l�se opgaverne p�, men dette
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