import random
import math
import modules_implementation
import vehicles_setup
import numpy as np

class PartitionBasedClustering:
    #dataset=list of all vehicles + RSU
    #cluster centers= all TAVs that generates tasks
    def __init__(self, dataset, cluster_centers):
        self.dataset = dataset
        self.max_nodes = len(dataset)
        self.clusters = list(cluster_centers)
        self.max_clusters = len(self.clusters)

    def partition_based_clustering(self, time, delta_time):
        #within delta_time euclidean distance should be always less than threshold radious
        #threshold is radious of vn (tav)
        candidate_sn = {}
        candidate_relay = {}
        sim_measure={}
        relay_node={}
        for c in self.clusters:
            candidate_sn[c.id]=[]
            candidate_relay[c.id]=[]
            sim_measure[c.id]=0
        for node in list(set(self.dataset)-set(self.clusters)):
            
            for cluster in self.clusters:
                flag=True
                for t in range(delta_time):
                    x1, y1, z1=node.get_curr_pos(time+t)
                    x2, y2, z2=cluster.get_curr_pos(time+t)
                    dist=modules_implementation.mobility_model(x1,x2,y1,y2,z1,z2)
                    if dist > cluster.radius:
                        flag=False
                        break
                if flag:
                    '''if cluster.id not in candidate_sn:
                        candidate_sn[cluster.id]=[]'''
                    if node in relay_node:
                        candidate_relay[cluster.id].append((node,dist))
                        if len(relay_node[node])==1:
                            candidate_relay[relay_node[node][0]].append((node,dist))
                        relay_node[node].append(cluster.id)

                    else:
                        relay_node[node]=[cluster.id]

                    candidate_sn[cluster.id].append((node,dist))
                    sim_measure[cluster.id]+=self.similarity_measure(dist)
                    
        return candidate_sn, candidate_relay, sim_measure,relay_node

    def similarity_measure(self, distance):
        return 1 / distance
    
    def print_sim_measure(sim_measure):
        for cluster_id in sim_measure:
            print("TAV id= ",cluster_id," density= ",len(sim_measure[cluster_id]))



class DensityBasedClustering:
    def __init__(self, candidate_sn, candidate_relay, sim_measure, relay_node):
        self.candidate_sn = candidate_sn
        self.candidate_relay = candidate_relay
        self.sim_measure = sim_measure
        self.relay_node = relay_node

    def density_based_clustering(self,n):
        for node in self.relay_node:
            clusterids=self.relay_node[node]
            mncluster,minid=n+1,0
            for id in clusterids:
                if(len(self.candidate_sn[id])<mncluster):
                    mncluster=len(self.candidate_sn[id])
                    minid=id
            #self.candidate_relay[minid]=list(filter(lambda x: x[0].id != node.id,self.candidate_relay[minid]))
            for idx,x in enumerate(self.candidate_relay[minid]):
                if x[0].id==node.id:
                    self.candidate_relay[minid].pop(idx)
                    break
            #print(node, clusterids, minid)
            self.relay_node[node]=[minid]
            for i in clusterids:
                #print(i)
                if(i!=minid):
                    print("---------------------------------------------------------------------------")
                    #print("before evaluation",i,self.relay_node)
                    #self.candidate_sn[i]=list(filter(lambda x: x[0].id != node.id, self.candidate_sn[i]))#doubt whether works
                    for idx,x in enumerate(self.candidate_sn[i]):
                        if x[0].id==node.id:
                            self.candidate_sn[i].pop(idx)
                            break
                    #self.relay_node[node].remove(i)
                    print("---------------------------------------------------------------------------")
                    #print("after evaluation",self.relay_node)
        return self.candidate_sn, self.candidate_relay,self.relay_node
    
    def print_cluster_density(self):
        for cluster_id in self.candidate_sn:
            print("TAV id= ",cluster_id," density= ",len(candidate_sn[cluster_id]))

'''
# Example usage:
candidate_sns = [(random.random(), random.random()) for _ in range(20)]
candidate_relays = [(random.random(), random.random()) for _ in range(20)]
cluster_heads = [(random.random(), random.random()) for _ in range(5)]
radius = 0.1  # Adjust the radius as needed
density_clustering = DensityBasedClustering(candidate_sns, candidate_relays, cluster_heads, radius)

for cluster_head in cluster_heads:
    core_objects = density_clustering.core_objects(cluster_head)
    density_connected_nodes = density_clustering.density_connected(core_objects, candidate_relays)
    noise_points = [node for node in candidate_sns if node not in core_objects]
    reachable_nodes = density_clustering.density_reachable(core_objects, density_connected_nodes, noise_points)

print("Reachable Nodes:", reachable_nodes)

print("Candidate SNs:")
for cluster_index, nodes in candidate_sn.items():
    print(f"Cluster {cluster_index + 1}: {nodes}")

print("\nCandidate Relay Nodes:")
for node in candidate_relay.keys():
    print(node)'''

def sort(node_dic, service=1):
    if service:
        for cluster_id in node_dic:
            node_dic[cluster_id].sort(key=lambda x:(x[0].computation_cycles_per_sec/x[1])) #x[1] stored final dist at the end of delta t from cluster center
    else:
        for cluster_id in node_dic:
            node_dic[cluster_id].sort(key=lambda x:(1/x[1]))

def estimated_latency(cluster_id, path, tav_lst):
    #print("path=",path)
    #path is a tuple where last node is the service node, 
    #dist from cluster to first node of the path
    #change cluster_id to corresponding TAV object as source
    source=None
    for tav in tav_lst:
        if(tav.id==cluster_id):
            source=tav
            break
    pos_src=source.get_curr_pos()
    pos_fst_node=path[0].get_curr_pos()
    dist_tav_1st_node_in_path=modules_implementation.mobility_model(pos_src[0],pos_fst_node[0],pos_src[1],pos_fst_node[1],pos_src[2],pos_fst_node[2])
    latency=0
    if(len(path)==1):
        latency=path[0].computation_cycles_per_sec/dist_tav_1st_node_in_path
    else:
        latency=1/dist_tav_1st_node_in_path
        pos_2nd_node=path[1].get_curr_pos()
        print(pos_2nd_node[0],pos_fst_node[0],pos_2nd_node[1],pos_fst_node[1],pos_2nd_node[2],pos_fst_node[2])
        dst_1st_2nd=modules_implementation.mobility_model(pos_2nd_node[0],pos_fst_node[0],pos_2nd_node[1],pos_fst_node[1],pos_2nd_node[2],pos_fst_node[2])
        latency+=(path[1].computation_cycles_per_sec/dst_1st_2nd)

    return latency

def sort_path(vo, tav_lst):
    for cluster_id in vo:
        print(vo[cluster_id])
        vo[cluster_id].sort(key=lambda path:estimated_latency(cluster_id,path,tav_lst))
    return vo

def find_path(candidate_sn, candidate_relay, relay_node, tav_lst):
    sort(candidate_sn)
    sort(candidate_relay,service=0)
    vo={}
    for cluster_id in candidate_sn:
        vo[cluster_id]=[]
        for sn in candidate_sn[cluster_id]:
            vo[cluster_id].append((sn[0],))
        for rn in candidate_relay[cluster_id]:
            #print("relay node: ",rn[0])
            cid_of_rn_as_sn=relay_node[rn[0]][0]
            #print(cluster_id, cid_of_rn_as_sn)
            for sni in candidate_sn[cid_of_rn_as_sn]:
                #print("serv node: ",sni[0])
                if(sni[0].id!=rn[0].id and modules_implementation.inrange(rn[0], sni[0])):
                    #print("in range...")
                    vo[cluster_id].append((rn[0],sni[0]))
    #print("vo=",vo)
    sort_path(vo, tav_lst)
    print("vo after sorting: ",vo)
    return vo
    

    
# Example Usage
if __name__ == "__main__":
    # Example data
    V={}
    tasks={}
    vehicles=vehicles_setup.main()
    for i, vehicle in enumerate(vehicles, start=1):
        print(f"Vehicle {i}: Id: {vehicle.id},Computation Cycles/s: {vehicle.computation_cycles_per_sec}, Transmission Power: {vehicle.transmission_power}, Radius: {vehicle.radius}, Position: {vehicle.get_curr_pos()}, Speed_Vector: {vehicle.speed_vec}")
    dataset=[]
    for i in vehicles:
        dataset.append(i)
    # dataset.append(len(vehicles)+1)
    Tavs=[]
    for i in range(5):
        ele=random.randint(0,14)
        while(dataset[ele] in Tavs):
            ele=random.randint(0,14)
        Tavs.append(vehicles[ele])
    pbclustering=PartitionBasedClustering(dataset,Tavs)
    candidate_sn, candidate_relay, sim_measure,relay_node=pbclustering.partition_based_clustering(0, 2)
    dbclustering=DensityBasedClustering(candidate_sn, candidate_relay, sim_measure,relay_node)
    candidate_sn,candidate_relay,relay_node=dbclustering.density_based_clustering(len(vehicles))
    #dbclustering.print_cluster_density()
    '''print("----------------------------------------------------------------------------------------------------------------")
    print(candidate_sn)
    print("---------------------------------------------------------------------------------------------------------------")
    print(candidate_relay)
    print("------------------------------------------------------------------------------------------------------------------")
    print(relay_node)'''
    # for vehicle in Tavs:
    #     vehicle.add_task(np.random.randint(500,  1001), 10, np.random.randint(2, 6))
    v0=find_path(candidate_sn, candidate_relay, relay_node, Tavs)

    for i in Tavs:
        print(modules_implementation.final_offloading_strategy(v0[i.id], i, i.task,vehicles,Tavs,candidate_sn,candidate_relay))

            


