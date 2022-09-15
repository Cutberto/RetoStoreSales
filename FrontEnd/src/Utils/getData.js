const data = [
  {
    date_x: "16/08/2017",
    sales: "4.274030181",
  },
  {
    date_x: "16/08/2017",
    sales: "0",
  },
  {
    date_x: "16/08/2017",
    sales: "3.495530483",
  },
  {
    date_x: "16/08/2017",
    sales: "2414.541711",
  },
  {
    date_x: "16/08/2017",
    sales: "0.438313589",
  },
  {
    date_x: "16/08/2017",
    sales: "396.4371786",
  },
  {
    date_x: "16/08/2017",
    sales: "17.14412362",
  },
  {
    date_x: "16/08/2017",
    sales: "824.2552908",
  },
  {
    date_x: "16/08/2017",
    sales: "820.6448919",
  },
  {
    date_x: "16/08/2017",
    sales: "141.2579654",
  },
  {
    date_x: "16/08/2017",
    sales: "162.9438196",
  },
  {
    date_x: "16/08/2017",
    sales: "154.6624425",
  },
  {
    date_x: "16/08/2017",
    sales: "3082.994584",
  },
  {
    date_x: "16/08/2017",
    sales: "31.76895942",
  },
  {
    date_x: "16/08/2017",
    sales: "2.585217108",
  },
  {
    date_x: "16/08/2017",
    sales: "33.8519548",
  },
];
export function getData() {
  let dates = [];
  let sales = [];
  console.log("data[0]", data[0]);
  for (var i = 0; i < data.length; i++) {
    dates.push(data[i].date_x);
    sales.push(data[i].sales);
  }
  let data2 = [dates, sales];
  return data2;
}
