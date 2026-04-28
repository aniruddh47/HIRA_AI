"""TML HIRA standard data extracted from the supplied internal PDF."""

from __future__ import annotations


HAZARDS: dict[str, str] = {
    "Gravity": "Fall, slip, trip, overturning of machine or trolleys.",
    "Machinery / Tool": "Moving parts, sharp edges, breakage, shock loading, flying objects, inadequate guards or vibration.",
    "Ergonomic": "Lifting, carrying, lowering, pulling, pushing, supporting, repetitive body movement or posture.",
    "Fire / Explosion": "Flammable or combustible material, gases, vapors, chemical reaction, smoke, conductive dust, static electricity or short circuit.",
    "Electrical": "Electrical shock, defective plugs, sockets, wires, cables, switches, exposed conductors or improperly protected instruments.",
    "Chemical": "Skin contact, inhalation, ingestion, oxygen deficiency, toxic gases or spillage.",
    "Confined Space": "Asphyxiation, explosion or restricted access.",
    "Biological": "Pathological organisms, biological fluids, animal bite or sting, biomedical waste.",
    "Pressure": "Pressure vessels, compressed air, gas, liquid or vacuum plant.",
    "Radiation": "Microwave, radio frequency, ultraviolet, infrared, laser, X-ray, ionizing radiation or welding radiation.",
    "Vehicular": "Traffic, cranes, forklifts, conveyors, jumbo equipment, overturning or malfunctioning.",
    "Heat & Temperature": "Hot surface, hot liquid, hot gas, flame, deep liquid vessels, sumps, welding spatters or cryogenic liquid.",
    "Environmental": "Heat, cold, extreme weather, noise, vibration, illumination, dust or fumes.",
    "Natural Calamity": "Building collapse, earthquake, fire after earthquake, flooding, cyclone, lightning or storm.",
    "Demographic": "Stampede, rioting, arson, theft, civil unrest, terrorism or sabotage.",
    "Human Factors / Behavioral issues": "Personal factors, behavior, vision, hearing, health conditions, horseplay or psychological imbalance.",
}


LIKELIHOOD_SCALE: dict[int, dict[str, str]] = {
    2: {"label": "Rare", "description": "Very low chance of occurrence."},
    3: {"label": "Occasional", "description": "Has or can happen once in 3 years."},
    4: {"label": "Probable", "description": "Has or can happen once in a year or leading to any over-riding criteria."},
    5: {"label": "Frequent", "description": "Has or can happen several times in a year."},
}


SCALE_OF_RISK: dict[int, dict[str, str]] = {
    2: {"label": "Only the work area", "description": "Welding area, grinding area, machine area or similar local work area."},
    3: {"label": "Within shop area", "description": "Entire shop shed, adjoining machines or process area is affected."},
    4: {"label": "Within plant boundary", "description": "Entire plant gets affected, production loss for many days or over-riding criteria applies."},
    5: {"label": "Outside plant boundary", "description": "External population or organizational reputation may get affected."},
}


LEVEL_OF_HARM: dict[int, dict[str, str]] = {
    2: {"label": "Insignificant", "description": "Momentary discomfort, inconvenience or minor near miss; no damage; exposure below prescribed standards."},
    3: {"label": "Harmful", "description": "Minor health effects or first aid case; moderate damage; exposure equal or above standards but less than 10%."},
    4: {"label": "Very Harmful", "description": "Medical treatment case, restricted work case, lost time injury or HIPO; severe damage; exposure within 10%-30% above standards."},
    5: {"label": "Extremely Harmful", "description": "Fatality, permanent disability, chronic or notifiable disease, major incident involving many people; severe environmental damage."},
}


PEOPLE_AFFECTED: dict[int, str] = {
    2: "0-2",
    3: "3-10",
    4: "11-100",
    5: "101 and above",
}


RISK_LEVELS: list[dict[str, object]] = [
    {
        "level": "Trivial",
        "min": 16,
        "max": 36,
        "acceptability": "Acceptable Risk",
        "action": "No action is required and no documentary record needs to be kept.",
    },
    {
        "level": "Tolerable",
        "min": 37,
        "max": 72,
        "acceptability": "Acceptable Risk",
        "action": "No additional controls are required. Consider an effective solution and monitor to ensure controls are maintained.",
    },
    {
        "level": "Moderate",
        "min": 73,
        "max": 130,
        "acceptability": "Unacceptable Risk",
        "action": "Efforts should be made to reduce the risk to acceptable level by using hierarchy of controls.",
    },
    {
        "level": "Substantial",
        "min": 131,
        "max": 250,
        "acceptability": "Unacceptable Risk",
        "action": "Work should not be started until the risk has been reduced. For work in progress, urgent action should be taken.",
    },
    {
        "level": "Intolerable",
        "min": 251,
        "max": 625,
        "acceptability": "Unacceptable Risk",
        "action": "Work should not be started or continued until risk has been reduced. If reduction is not possible, work remains prohibited.",
    },
]


OVERRIDING_CRITERIA: dict[str, str] = {
    "DC": "Domino Concern: operations/processes that can trigger or multiply risk and create potential emergency conditions.",
    "LC": "Legislative Concern: OHS hazards or risks covered by existing and applicable EHS legislation and notifications.",
    "E": "Emergency: unplanned event that may threaten life, health or property on a large scale and require specialist/external support.",
}


CONTROL_HIERARCHY = [
    "Elimination",
    "Substitution",
    "Engineering controls",
    "Administrative controls",
    "PPE",
]


HIRA_STEPS = [
    "Preparation",
    "Hazard Identification",
    "Risk Assessment",
    "Plan Control Measures",
    "Record Keeping",
    "Implementation and Review",
]


REVIEW_TRIGGERS = [
    "incident, accident or ill health",
    "statutory regulation amendment or new regulation",
    "interested party concern",
    "audit observation",
    "new process, equipment, storage facility, installation or change",
    "product, activity or service change",
    "new product or model development",
    "change in existing control",
    "process or layout change",
    "raw material substitution or chemical change",
    "SOP, SMP or WIS change",
    "Management of Change",
    "completion of OHS objectives and programme",
]
