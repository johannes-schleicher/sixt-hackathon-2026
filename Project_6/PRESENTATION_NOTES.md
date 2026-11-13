# Präsentation – Stichpunkte (Rent me if you can · Project 6)

## 1. Aufgabe
- Pro zurückgegebenem Auto vorhersagen: Dauer jedes einzelnen Tasks und wann das Auto insgesamt fertig ist.
- Hauptmetrik ist der Fehler pro Task (WAPE). Daneben zählen der Fehler der Gesamtzeit, Plausibilität und Reproduzierbarkeit.
- Ziel fürs Geschäft: dem Kunden sagen können, wann sein Auto fertig ist.

## 2. Wie läuft eine Filiale wirklich? (Struktur aus den Daten)
- Das Laden von Elektroautos läuft unbeaufsichtigt und parallel zu allem anderen.
- Die übrigen Tasks werden auf die anwesenden Mitarbeiter verteilt, der längste Task zuerst.
- Bei einer Reparatur gilt eine feste Kette: Schadenscheck → Reparatur → Außenwäsche.
- Gearbeitet wird nur während der Öffnungszeiten. Was nicht fertig wird, geht über Nacht oder übers Wochenende.
- Schlüsselbox: Autos, die außerhalb der Öffnungszeiten abgegeben werden, warten 12 Min. pro Auto, das vor ihnen in der Box liegt (Korrelation 0,998).
- Vor der Arbeit kommt die Warteschlange. Diese Wartezeit lernt ein eigenes Modell aus der Auslastung: wie viele Autos in den letzten 15 bis 120 Min. kamen und wie viel Arbeit sie mitbringen.

## 3. Was die einzelnen Tasks treibt (erklärbare Modelle)
- Pro Task ein einfaches lineares Modell (Ridge-Regression), jeder Effekt ist in Minuten ablesbar.
- Kindersitz: +7,5 Min. Innenreinigung. Raucherauto: +7,6 Min. Spezialreinigung.
- Verschmutzung, Regen und Transporter machen die Außenwäsche länger.
- Die Ladedauer hängt am Akkustand (9 bis 40 Min.).
- Reifencheck: etwa 45 Min. mit einer Person, 18 Min. an Flughäfen.
- Gemeldeter Schaden: etwa +18 bis 25 Min. Reparatur.
- Ergebnis: Fehler pro Task (WAPE) von 0,55 auf **0,075**. Im README gilt unter 0,2 als „stark“.

## 4. Getestet und bewusst verworfen
- Feiertage und Jahreszeit haben in den Daten keinen Effekt, deshalb nicht im Modell.
- Wetter wirkt nur bei der Außenwäsche.
- Reparatur und Reifencheck zusätzlich aus den Trainingsdaten lernen: kein Gewinn. Die Reparatur streut zufällig um etwa ±16 Min.
- Autos kurz vor Ladenschluss aus der Bewertung herausnehmen: nicht erlaubt, deshalb offen ausgewiesen statt versteckt.

## 5. Entwicklung des Fehlers der Gesamtzeit
- Baseline 99 Min. → 14,7 → 13,3 → 12,3 → **8,1 Min.**
- Der große Sprung kam durch das Modell für die Wartezeit (Gradient Boosting). Die Grundidee stammt aus einer Codex-Variante, die verglichen und übernommen wurde.
- Plausibilitätsregel: Kein Auto ist früher fertig als seine eigene Arbeitszeit. Vorher gab es 220 solche Fälle, jetzt null.

## 6. Ehrliche Ergebnisse
- Getestet nur auf Tagen und Wochen, die das Modell nie gesehen hat: jeder Tag der Validierungswoche einzeln herausgenommen, dazu 3 spätere Trainingswochen.
- Fehler der Gesamtzeit: **8,1 Min.** (README „gut“: unter 15).
- Median-Fehler 2,3 Min., **91 %** aller Autos innerhalb von 10 Min.
- Spätere Wochen: 7,5 / 6,0 / 6,2 Min.
- Reproduzierbar: zwei Durchläufe liefern identische Werte, etwa 2 Sekunden auf dem Laptop, Paketversionen fest vorgegeben.

## 7. Was der Kunde hört
- Rückgabe tagsüber (88 % der Autos): „Fertig um 14:35.“ Im Schnitt **4,3 Min.** daneben.
- Rückgabe kurz vor Ladenschluss (12 %): „Morgen ab Öffnung fertig.“ Das richtige Datum stimmt in 97 % der Fälle.
- Diese 12 % machen etwa die Hälfte des Gesamtfehlers aus, sind am Abend aber ohnehin nicht mehr vermietbar.

## 8. Grenzen und Ausblick
- Montag bis Freitag und Sonntag: etwa 3,5 bis 9 Min. Samstag: etwa 21 Min., wegen der kurzen Öffnungszeit und der Übergänge aufs Wochenende.
- Nächster Schritt: eine „fertig bis“-Zusage, die in 9 von 10 Fällen hält.
- Mit dem Modell lassen sich zusätzliche Mitarbeiter für den Abendansturm planen.

**Schlusssatz:** „Wir haben nicht einfach ein Black-Box-Modell trainiert, sondern zuerst herausgefunden, wie eine Filiale arbeitet, und genau das nachgebaut.“
