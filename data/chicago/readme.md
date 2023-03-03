1. `all_data.pkl`
   2016.2~2016.9，one time interval is 1h
   shape(T, D, W, H),D=41
   ************
   0:risk
   1~24:time_period，(one-hot)
   25~31:day_of_week，(one-hot)
   32:holiday，(one-hot)    
   ************
   33:temperature
   34:Clear,(one-hot)
   35:Cloudy，(one-hot)
   36:Rain，(one-hot)
   37:Snow，(one-hot)
   38:Mist，(one-hot)
   ******************
   39:inflow
   40:outflow
   ***********

2. `risk_mask.pkl`
   shape(W,H)
   top risk region mask

3. `risk_adj.pkl`
   risk similarity graph adjacency matrix
   shape (N,N)
   
4. `road_adj.pkl`
   road similarity graph adjacency matrix
   shape(N,N)

5. `grid_node_map.pkl`
   map graph data to grid data
   shape (W*H,N)


