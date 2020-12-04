import csv

def loadWings(fname):
    with open(fname) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
        rows = []
        for row in reader:
            rows.append(row)

        def lerp(field, alpha):
            last = rows[0]
            for row in rows:
                if row['Alpha'] > alpha:
                    break
                last = row

            if last == row:
                return row[field]
            else:

                lerpValue = (alpha - last['Alpha']) / (row['Alpha'] - last['Alpha'])
                return last[field] + (row[field] - last[field]) * lerpValue

        def Cl(alpha):
            return lerp('Cl', alpha)

        def Cd(alpha):
            return lerp('Cd', alpha)

        def Cm(alpha):
            return lerp('Cm', alpha)

        return Cl, Cd, Cm