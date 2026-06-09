# Einfache CO2 Kreislaufberechnung

[**Zur Anwendung**](https://co2berechnung.streamlit.app/)

## Funktionen

- Berechnung zentraler Systemdaten wie Kältemittelmassenstrom und EER.
- Berechnung von neun Kreislaufpunkten mit spezifischer Enthalpie, Entropie und Dichte.
- Es wird subkritische und transkritische Prozessführung unterstützt.
- Auslegung von jeweils zwei Rohrleitungsvarianten für Heißgasleitung, Flüssigkeitsleitung und Saugleitung inklusive Innenvolumen und Mantelfläche.
- Export aller berechneten Daten als CSV-Datei.

## Eingabeparameter

Folgende Eingaben stehen in der App zur Verfügung:

- Projektname
- Eingabemodus
- Kälteleistung, Wärmeleistung oder Verdichtervolumenstrom, abhängig vom gewählten Modus
- Gaskühlerdruckbestimmung
- Gaskühleraustrittstemperatur
- Gaskühlerdruck
- Abkühlung nach dem Gaskühler
- Verdampfungstemperatur
- Verdampferüberhitzung
- Saugleitungsüberhitzung
- Aktivierung der Rohrleitungsdimensionierung
- Leitungslängen für Heissgas-, Flüssigkeits- und Saugleitung

## CO2-spezifische Berechnung

Die Anwendung verwendet **CO2** als festes Arbeitsmedium und bestimmt den Prozessbereich anhand des Hochdruckniveaus automatisch als **subkritisch** oder **transkritisch**.

Der Gaskühlerdruck kann entweder automatisch über die hinterlegte Näherungsfunktion bestimmt oder manuell vorgegeben werden.

Zusätzlich wird der erkannte Prozessbereich in den **Systemdaten** ausgegeben.

## Kreislaufpunkte

Folgende Kreislaufpunkte werden berechnet:
- **1:** Verdichtereingang nach Überhitzung von Verdampfer und Saugleitung
- **2:** Verdichterausgang und Gaskühlereintritt
- **3:** Gaskühleraustritt und Expansionsventileingang
- **4:** Expansionsventilausgang und Verdampfereintritt
- **5:** Verdampferausgang und Eingang Saugleitung
- **0''**: Gesättigtes Gas bei Verdampfungsdruck
- **0'**: Flüssigkeit bei Verdampfungsdruck

## Rohrleitungsdimensionierung

Die Rohrleitungsberechnung verwendet hinterlegte Kupferrohrabmessungen und bewertet je Leitung Strömungsgeschwindigkeit, Jacobs-Geschwindigkeit, Druckverlust, Außenmantelfläche und Innenvolumen.

Für jede Leitung werden zwei Varianten bestimmt:

- eine Variante mit minimalem Druckverlust
- eine Variante mit minimalem Durchmesser

## CSV-Export

Über den Button **„CSV-Datei erstellen“** können die Daten exportiert werden.

## Technische Basis

Die Anwendung basiert auf **Streamlit** für die Oberfläche, **CoolProp** für Stoffdaten und thermodynamische Zustandsgrößen, **NumPy** und **Pandas** für Berechnung und Datenaufbereitung sowie **SciPy** für numerische Lösungsverfahren mit `brentq`.
