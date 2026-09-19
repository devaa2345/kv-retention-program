"""Unit tests for the natural-text scorer, on constructed generations, run before any verdict."""
from p3.natural import facts as F
from p3.natural.score import score_instance, score_one

f = F.Fact("F00", "Halworth", "Marisa Enders", "Department of Fisheries", "2019", "41,200",
           "Ravensmoor")
others = ["Tobias Krell", "Helena Quist", "Gareth Onslow"]
persons = ["Marisa Enders"] + others
E1, E3, E5 = (F.element_answers(f, lv) for lv in (1, 3, 5))
T = []


def ok(name, got, want):
    T.append((name, got == want))
    if got != want:
        print("FAIL", name, "got", got, "want", want)


ok("exact level 1", score_one("Marisa Enders", E1, persons), 1.0)
ok("case/space", score_one("  marisa   ENDERS.", E1, persons), 1.0)
ok("sentence answer", score_one("It was completed by Marisa Enders.", E1, persons), 1.0)
ok("surname only is not the person", score_one("Enders", E1, persons), 0.0)
ok("wrong person", score_one("Tobias Krell", E1, persons), 0.0)
ok("shotgun list of all names", score_one("Marisa Enders, Tobias Krell, Helena Quist", E1, persons), 0.0)
ok("empty", score_one("", E1, persons), 0.0)
ok("level 3 exact", score_one("Marisa Enders | Department of Fisheries | 2019", E3, persons), 1.0)
ok("level 3 missing year", score_one("Marisa Enders | Department of Fisheries", E3, persons), 0.0)
ok("level 3 wrong dept", score_one("Marisa Enders | Department of Ports | 2019", E3, persons), 0.0)
ok("year boundary: 2019 not inside 12019", score_one("Marisa Enders | Department of Fisheries | 12019", E3, persons), 0.0)
ok("year boundary: 2019 not inside 20195", score_one("Marisa Enders | Department of Fisheries | 20195", E3, persons), 0.0)
ok("level 5 exact with $ and commas", score_one(
    "Marisa Enders | Department of Fisheries | 2019 | $41,200 | Ravensmoor", E5, persons), 1.0)
ok("level 5 no commas", score_one(
    "Marisa Enders | Department of Fisheries | 2019 | 41200 | Ravensmoor", E5, persons), 1.0)
ok("level 5 amount substring 4120 wrong", score_one(
    "Marisa Enders | Department of Fisheries | 2019 | $4,120 | Ravensmoor", E5, persons), 0.0)
ok("level 5 amount 141200 wrong", score_one(
    "Marisa Enders | Department of Fisheries | 2019 | $141,200 | Ravensmoor", E5, persons), 0.0)
ok("exemplar echo scores 0", score_one(
    "Dara Quill | Department of Ports | 2011 | $20,500 | Alderney", E5, persons), 0.0)
ok("exemplar echo level 1", score_one("Dara Quill", E1, persons), 0.0)
ok("prefix garbage does not hide match", score_one("A: Marisa Enders", E1, persons), 1.0)
ok("name inside longer word rejected", score_one("Marisa Endersby", E1, persons), 0.0)
NL = chr(10)
ok("correct then invented B record", score_one(
    "A: Marisa Enders | Department of Fisheries | 2019" + NL * 2 + "B: Tobias Krell | Department of Lands | 2015" + NL * 2,
    E3, persons), 1.0)
ok("correct then single-newline B record", score_one(
    "A: Marisa Enders | Department of Fisheries | 2019" + NL + "B: Tobias Krell | Department of Lands | 2015",
    E3, persons), 1.0)
ok("wrong first block then correct B still 0", score_one(
    "A: Tobias Krell | Department of Lands | 2015" + NL * 2 + "B: Marisa Enders | Department of Fisheries | 2019",
    E3, persons), 0.0)
ok("multi-line numbered answer kept", score_one(
    NL.join(["1. Marisa Enders |", "2. Department of Fisheries |", "3. 2019"]), E3, persons), 1.0)
ok("multi-line prose answer kept", score_one(
    NL.join(["The person is Marisa Enders.", "The department is the Department of Fisheries.",
             "The year is 2019."]), E3, persons), 1.0)
ok("blank-line-separated correct fields are cut after first block", score_one(
    "Marisa Enders" + NL * 2 + "Department of Fisheries | 2019", E3, persons), 0.0)
vs = [dict(elements=E1)] * 4
ok("instance mean 2/4", score_instance(["Marisa Enders", "x", "Marisa Enders", ""], vs, persons), 0.5)
ok("instance mean is not 'any of H'", score_instance(["Marisa Enders", "x", "x", "x"], vs, persons), 0.25)
try:
    score_instance(["a"], vs, persons)
    ok("length mismatch raises", False, True)
except ValueError:
    ok("length mismatch raises", True, True)
print("%d/%d passed" % (sum(g for _, g in T), len(T)))
raise SystemExit(0 if all(g for _, g in T) else 1)
