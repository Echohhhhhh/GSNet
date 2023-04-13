1. `all_data.pkl`
   2013.1~2013.12，one time interval is 1h

   shape(T, D, W, H),D=48

   0:risk

   1~24:time_period，(one-hot)

   25~31:day_of_week，(one-hot)

   32:holiday，(one-hot)

   33~39:POI

   40:temperature

   41:Clear,(one-hot)

   42:Cloudy，(one-hot)

   43:Rain，(one-hot)

   44:Snow，(one-hot)

   45:Mist，(one-hot)

   46:inflow

   47:outflow

2. `risk_mask.pkl`
   shape(W,H)
   top risk region mask

3. `risk_adj.pkl`
   risk similarity graph adjacency matrix
   shape (N,N)
   
4. `road_adj.pkl`
   road similarity graph adjacency matrix
   shape(N,N)

5. `poi_adj.pkl`
   poi similarity graph adjacency matrix
   shape(N,N)
   
6. `grid_node_map.pkl`
   map graph data to grid data
   shape (W*H,N)

