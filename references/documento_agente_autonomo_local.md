# Agente Autónomo Local – Documento de Objetivos y Alcance

## 1. Objetivo general

El objetivo de este proyecto es diseñar e implementar un **agente autónomo** que se ejecute de forma persistente en un servidor propio y sea capaz de **interpretar instrucciones**, **decidir acciones** y **ejecutar tools simples de manera segura**, sin intervención humana directa en cada paso.

El agente debe actuar como una **capa de automatización inteligente**, similar a un operador técnico junior, pero con límites estrictos de seguridad y alcance.

---

## 2. Qué queremos lograr

Queremos construir un sistema que:

- Corra **24/7 en nuestro servidor** (VPS, bare metal o local).
- Exponga una **API** para enviarle instrucciones (texto o JSON).
- Use un **modelo LLM** para razonar qué hacer.
- Pueda **invocar tools predefinidas** (no comandos arbitrarios).
- Devuelva resultados estructurados y auditables.
- Sea **extensible**, para agregar nuevas tools sin rediseñar todo.

En resumen: un agente confiable, controlado y útil para tareas operativas reales.

---

## 3. Qué NO queremos

Es importante definir límites claros:

- ❌ No queremos un shell libre tipo "ejecutá cualquier comando".
- ❌ No queremos que el LLM tenga permisos de root.
- ❌ No queremos que el agente pueda acceder a todo el filesystem.
- ❌ No queremos lógica mágica u opaca sin logs.
- ❌ No queremos depender de una UI compleja para el MVP.

El foco está en **control, previsibilidad y seguridad**.

---

## 4. Casos de uso principales

Ejemplos concretos de lo que el agente debería poder hacer:

- Consultar el estado del servidor (CPU, RAM, disco).
- Listar archivos en directorios permitidos.
- Leer logs o archivos de configuración específicos.
- Ejecutar jobs predefinidos (deploy, build, backup, restart controlado).
- Consultar APIs externas autorizadas.
- Responder con diagnósticos o resúmenes.

Todos estos casos pasan **siempre** por tools explícitas.

---

## 5. Arquitectura conceptual

### Componentes principales

1. **Agent API**  
   Servicio HTTP (por ejemplo FastAPI) que recibe instrucciones y devuelve respuestas.

2. **LLM Connector**  
   Adaptador al modelo elegido (remoto o local) encargado del razonamiento.

3. **Tool Registry**  
   Catálogo de tools disponibles, con:
   - nombre
   - descripción
   - esquema de entrada (JSON Schema)
   - permisos

4. **Tool Executor**  
   Capa que ejecuta realmente las tools, aplicando:
   - validación de inputs
   - timeouts
   - límites de recursos
   - logging

5. **Storage / Auditoría**  
   Persistencia de:
   - requests
   - decisiones del agente
   - tools ejecutadas
   - outputs y errores

---

## 6. Qué es una tool en este sistema

Una **tool** es una acción controlada, explícita y validada.

Características:

- Tiene un propósito claro.
- Acepta solo inputs tipados.
- Opera en un entorno restringido.
- Devuelve un resultado estructurado (JSON).
- Puede fallar de forma segura.

Ejemplos:

- `health_check()`
- `list_dir(path)`
- `read_file(path)`
- `run_deploy(project)`

El LLM **no ejecuta código**, solo **elige** qué tool usar.

---

## 7. Seguridad como requisito base

El diseño parte de la premisa: *el agente es potencialmente peligroso si no se limita*.

Medidas mínimas obligatorias:

- Allowlist de tools (no ejecución libre).
- Usuario del sistema sin privilegios.
- Directorios aislados (workspace).
- Validación fuerte con schemas.
- Logs completos y revisables.
- Timeouts y límites de output.

La seguridad no es un extra, es parte del core.

---

## 8. MVP propuesto

### Alcance inicial

- API única: `POST /agent/run`
- Input: mensaje en texto
- Tools iniciales:
  - `health_check`
  - `list_dir`
  - `read_file`
- Persistencia en SQLite
- Autenticación por API Key

### Objetivo del MVP

Demostrar que:

- El agente puede razonar correctamente.
- Las tools se ejecutan de forma segura.
- El sistema es observable y confiable.

---

## 9. Evolución futura (no MVP)

- Cola de jobs para tareas largas.
- Sandbox por tool (Docker / Firecracker).
- Memoria a largo plazo.
- Policies dinámicas de permisos.
- UI web para monitoreo.
- Múltiples agentes especializados.

---

## 10. Resultado esperado

Al finalizar este proyecto tendremos:

- Un agente autónomo útil en producción.
- Capaz de operar infraestructura básica.
- Extensible y mantenible.
- Seguro por diseño.

Este documento sirve como **base de alineación técnica** y guía para la implementación.
