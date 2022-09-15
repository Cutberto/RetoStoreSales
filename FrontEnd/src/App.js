import React, { useEffect, useState } from "react";
import Navbar from "./Component/NavBar";
import { Card } from "react-bootstrap";
import "./App.css";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Button from "react-bootstrap/Button";
import Alert from "react-bootstrap/Alert";
import GraficaLienal from "./Component/ModeloLineal";
import GraficaForest from "./Component/ModeloForest";

const App = (props) => {
  return (
    <Router>
      <Routes>
        <Route
          path="/lineal"
          element={
            <div className="App" style={{ height: "100%", width: "100%" }}>
              <div>
                <Navbar />
              </div>
              <div
                style={{
                  display: "flex",
                  justifyContent: "center",
                }}
              >
                <Card style={{ height: "1000px", width: "1000px" }}>
                  <Card.Header>Gráfica de ventas</Card.Header>
                  <GraficaLienal
                    style={{ height: "1000px", width: "1000px" }}
                  />
                </Card>
              </div>
            </div>
          }
        />

        <Route
          path="/forest"
          element={
            <div className="App" style={{ height: "100%", width: "100%" }}>
              <div>
                <Navbar />
              </div>
              <div
                style={{
                  display: "flex",
                  justifyContent: "center",
                }}
              >
                <Card style={{ height: "1000px", width: "1000px" }}>
                  <Card.Header>Gráfica de ventas</Card.Header>
                  <GraficaForest
                    style={{ height: "1000px", width: "1000px" }}
                  />
                </Card>
              </div>
            </div>
          }
        />

        <Route
          path="/"
          element={
            <div className="App" style={{ height: "100%", width: "100%" }}>
              <div>
                <Navbar />
              </div>
              <div
                style={{
                  display: "flex",
                  justifyContent: "center",
                }}
              >
                <Card style={{ height: "1000px", width: "1000px" }}>
                  <Card.Header>Gráfica de ventas</Card.Header>
                  <Card.Body>
                    <Alert variant="primary">
                      El modelo lineal se ejecuta muy rápido, sin embargo su
                      precisión es más baja que otros modelos.
                    </Alert>
                    <Button variant="primary" href="/lineal">
                      Ejecutar Modelo Lineal
                    </Button>{" "}
                    <br /> <br />
                    <Alert variant="success">
                      El modelo de random forest es más lento, su ejecución toma
                      un par de minutos pero tiene una mayor precisión en sus
                      predicciones
                    </Alert>
                    <Button variant="success" href="/forest">
                      Ejecutar Modelo de Random Forest
                    </Button>{" "}
                  </Card.Body>
                </Card>
              </div>
            </div>
          }
        />
      </Routes>
    </Router>
  );
};

export default App;
