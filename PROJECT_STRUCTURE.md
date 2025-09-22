# LEAP-PSW Projektstruktur

## Ordnerstruktur Übersicht

```
leap-PSW/
├── src/                            # Quellcode
│   ├── backend/                    # Backend-Services (Microservices)
│   │   ├── api-gateway/           # API Gateway Service
│   │   ├── auth-service/          # Authentifizierung & Autorisierung
│   │   ├── user-service/          # Benutzerverwaltung
│   │   ├── workflow-service/      # Workflow-Engine
│   │   ├── document-service/      # Dokumentenmanagement
│   │   ├── notification-service/  # Benachrichtigungen
│   │   ├── audit-service/         # Audit-Logging
│   │   ├── reporting-service/     # Reporting & Analytics
│   │   ├── integration-service/   # Externe Integrationen
│   │   └── common/                # Gemeinsame Backend-Komponenten
│   │       ├── utils/             # Utility-Funktionen
│   │       ├── middleware/        # Express/Fastify Middleware
│   │       └── models/            # Gemeinsame Datenmodelle
│   │
│   ├── frontend/                   # Frontend-Anwendungen
│   │   ├── web-portal/            # Hauptwebanwendung (Next.js)
│   │   │   ├── components/        # React-Komponenten
│   │   │   ├── pages/            # Next.js Pages
│   │   │   ├── services/         # API-Services
│   │   │   ├── hooks/            # Custom React Hooks
│   │   │   ├── utils/            # Utility-Funktionen
│   │   │   └── styles/           # CSS/SCSS Styles
│   │   ├── admin-dashboard/       # Admin-Dashboard (React)
│   │   │   ├── components/        # React-Komponenten
│   │   │   ├── pages/            # Seiten-Komponenten
│   │   │   └── services/         # API-Services
│   │   └── mobile-app/            # Mobile App (React Native)
│   │       ├── screens/          # App-Screens
│   │       ├── components/       # React Native Komponenten
│   │       └── services/         # API-Services
│   │
│   └── shared/                     # Gemeinsamer Code
│       ├── types/                 # TypeScript Type Definitionen
│       ├── constants/             # Konstanten
│       ├── validators/            # Validierungs-Schemas
│       └── interfaces/            # Gemeinsame Interfaces
│
├── infrastructure/                 # Infrastructure as Code
│   ├── docker/                    # Docker-Konfigurationen
│   ├── kubernetes/                # Kubernetes Manifests
│   └── terraform/                 # Terraform-Konfigurationen
│
├── database/                       # Datenbank-bezogen
│   ├── migrations/                # Datenbank-Migrationen
│   └── seeds/                     # Seed-Daten für Entwicklung
│
├── config/                        # Konfigurationsdateien
│   ├── development/               # Entwicklungsumgebung
│   ├── staging/                   # Staging-Umgebung
│   └── production/                # Produktionsumgebung
│
├── docs/                          # Dokumentation
│   ├── api/                       # API-Dokumentation
│   ├── architecture/              # Architektur-Dokumentation
│   └── deployment/                # Deployment-Guides
│
├── tests/                         # Tests
│   ├── unit/                      # Unit-Tests
│   ├── integration/               # Integrationstests
│   └── e2e/                       # End-to-End Tests
│
├── scripts/                       # Build- und Utility-Scripts
├── tools/                         # Entwickler-Tools
│   ├── cli/                       # CLI-Tools
│   └── migration/                 # Migrations-Tools
│
├── monitoring/                    # Monitoring-Konfigurationen
│   ├── grafana/                   # Grafana Dashboards
│   ├── prometheus/                # Prometheus Konfiguration
│   └── elk/                       # ELK Stack Konfiguration
│
└── .github/                       # GitHub-spezifisch
    └── workflows/                 # GitHub Actions Workflows
```

## Beschreibung der Hauptordner

### `/src/backend/`
Enthält alle Microservices des LEAP-PSW Systems. Jeder Service ist eigenständig und kommuniziert über REST APIs oder Message Queues.

### `/src/frontend/`
Drei separate Frontend-Anwendungen:
- **web-portal**: Hauptanwendung für Endbenutzer
- **admin-dashboard**: Verwaltungsoberfläche für Administratoren
- **mobile-app**: Native Mobile-Anwendung

### `/src/shared/`
Code, der zwischen Frontend und Backend geteilt wird (TypeScript-Typen, Validierungslogik, etc.)

### `/infrastructure/`
Infrastructure as Code für verschiedene Deployment-Szenarien (Docker für Entwicklung, Kubernetes für Produktion, Terraform für Cloud-Ressourcen)

### `/docs/`
Vollständige Projektdokumentation, aufgeteilt nach API-Spezifikationen, Architektur-Entscheidungen und Deployment-Anleitungen

### `/tests/`
Strukturierte Testsuites für verschiedene Testebenen

### `/monitoring/`
Konfigurationen für das Monitoring-Stack (Grafana, Prometheus, ELK)

## Nächste Schritte

1. Initialisierung der Package-Manager in den jeweiligen Ordnern
2. Erstellung der grundlegenden Konfigurationsdateien
3. Setup der Docker-Entwicklungsumgebung
4. Implementierung der ersten Services