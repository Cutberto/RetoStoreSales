import { DateTimeTickStrategy } from "@arction/lcjs";
import React from "react";
import Chart from "react-apexcharts";
import { getData } from "../Utils/getData";
import { useState, useEffect } from "react";
import Spinner from "react-bootstrap/Spinner";

export default function GraficaLineal() {
  const [sales, setSales] = useState("");
  const [dates, setDates] = useState("");
  const [json, setJson] = useState("");
  const [isLoading, setLoading] = useState(true);

  useEffect(async () => {
    setLoading(true);
    await getJson();
    setLoading(false);
  }, []);

  const getJson = async () => {
    console.log("uwu");
    const response = await fetch("http://192.168.100.39/lineal", {
      method: "GET",
      headers: new Headers({ "Content-type": "application/json" }),
      mode: "cors",
    });
    let jsonData = await response.json();
    console.log(jsonData);
    setJson(jsonData);

    let dates = [];
    let sales = [];

    jsonData = JSON.parse(jsonData);
    console.log(jsonData);

    for (var fecha in jsonData) {
      console.log(fecha, jsonData[fecha]);
      console.log(fecha);
      let temp_fecha = await new Date();
      temp_fecha.setTime(fecha);
      let fecha_str =
        temp_fecha.getDate() +
        "/" +
        (temp_fecha.getMonth() + 1) +
        "/" +
        temp_fecha.getFullYear();

      console.log("tempfecha:", temp_fecha);
      console.log("fecha_str", fecha_str);
      dates.push(fecha_str);
      sales.push(jsonData[fecha]);
    }
    setSales(sales);
    setDates(dates);
    console.log("listo");
  };

  console.log("sales:", sales);

  console.log("dates:", dates);

  const guestSeries = [
    {
      name: "Sales",
      data: sales,
    },
  ];
  const guestOption = {
    chart: {
      id: "guest",
      group: "social",
      animations: {
        speed: 100,
      },
    },
    xaxis: {
      title: {
        text: "Días",
      },
      categories: dates,
    },
    yaxis: {
      title: {
        text: "Artículos vendidos",
      },
    },
    stroke: {
      curve: "smooth",
    },
  };

  if (isLoading) {
    return (
      <div>
        Entrenando modelo c:
        <br />
        <br />
        <Spinner animation="border" role="status">
          <span className="visually-hidden">Loading...</span>
        </Spinner>
      </div>
    );
  } else {
    return (
      <div>
        <Chart
          type="line"
          series={guestSeries}
          options={guestOption}
          width="1000"
          height="400"
        />
      </div>
    );
  }
}
