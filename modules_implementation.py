import math
import vehicles_setup
import numpy as np
# import clustering
import random

#returns euclidian distance between two vehicles
def mobility_model(x1,x2,y1,y2,z1,z2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return distance

def path_loss(distance, velocity_vector_vn):
    num_samples = 1000  # Number of Monte Carlo samples
    #velocity_vector_vn = np.array([v_x_prime, v_y_prime, v_z_prime])  # Velocity vector of vehicle v_n_prime
    frequency = 2.4e9  # Frequency in Hz
    c = 3e8  # Speed of light in m/s


    # Calculate time delay based on distance and speed
    time_delay = distance / np.linalg.norm(np.asarray(velocity_vector_vn))

    # Generate channel gain samples using Rayleigh fading model
    gamma_n_m = np.random.rayleigh(scale=np.sqrt(1/2), size=num_samples)

    # Apply path loss model (e.g., Friis transmission equation)
    path_loss = (c / (4 * np.pi * frequency * distance)) ** 2

    # Apply fading effects and path loss to calculate channel gain
    channel_gain_samples = gamma_n_m / path_loss

    # Calculate average channel gain
    average_channel_gain = np.mean(channel_gain_samples)

    print("Average channel gain from v_n' to v_m:", average_channel_gain)
    return average_channel_gain




# # Example usage:
# '''distance = 100  # meters
# frequency = 2.4e9  # 2.4 GHz
# tx_power = 20  # dBm
# channel_gain = path_loss(distance, frequency, tx_power)
# print("Channel Gain:", channel_gain, "dB")'''

def inrange(vk,vt):
    vkpos=vk.get_curr_pos()
    vtpos=vt.get_curr_pos()
    dist=mobility_model(vkpos[0],vtpos[0],vkpos[1],vtpos[1],vkpos[2],vtpos[2])
    if dist<vt.radius:
        return 1
    return 0

def isrelay(v_n,v_m,candidate_rn):
    for ele in candidate_rn[v_n.id]:
        if(ele[0].id==v_m.id):
            return 1
    return 0

def signal_to_noise_ratio(vn,vm,vehicles):#add vehuicles list a sa paramter if something is wrong
    # global vehicles
    vnpos=vn.get_curr_pos()
    vmpos=vm.get_curr_pos()
    print("------------------------------------------------------------------------------------------------------------------")
    print("vn pos is",vnpos)
    print("------------------------------------------------------------------------------------------------------------------")
    print("vmpos is",vmpos)
    nmdistance=mobility_model(vnpos[0],vmpos[0],vnpos[1],vmpos[1],vnpos[2],vmpos[2])
    print("distance is ",nmdistance)
    nm_channel_gain = path_loss(nmdistance, vn.get_speed_vector())
    denominator=0
    for vk in vehicles:
        if vk.id!=vn.id:
            if inrange(vk,vm):
                vkpos=vk.get_curr_pos()
                km_dist=mobility_model(vmpos[0],vkpos[0],vmpos[1],vkpos[1],vmpos[2],vkpos[2])
                cg = path_loss(km_dist, np.array(vk.get_speed_vector()))
                denominator+=(cg* vk.transmission_power)
    denominator+=(-100)**2
    snr= (nm_channel_gain * vn.transmission_power)/denominator
    print("snr= ",snr)
    return snr


def shannon_capacity(bandwidth, snr):
    """
    Calculate the channel capacity using Shannon's formula.
    
    Parameters:
        bandwidth (float): Bandwidth of the channel in Hertz (Hz).
        snr (float): Signal-to-noise ratio.
    
    Returns:
        float: Channel capacity in bits per second (bps).
    """
    capacity = bandwidth * math.log2(1 + snr)
    return capacity

def get_transmission_rate(v1, v2,vehicles):
    bandwidth = 1e6  # 1 MHz bandwidth
    #snr = 10  # 

    #transmission_rate = shannon_capacity(bandwidth, 10**(snr/10))
    snr=signal_to_noise_ratio(v1,v2,vehicles)
    transmission_rate = shannon_capacity(bandwidth, snr)
    print("Transmission Rate:", transmission_rate, "bps")
    return transmission_rate




def local_computing_latency(v_n,t_n):
    d_loc_n=t_n[0]
    f_loc_n=v_n.computing_capacity
    c_loc_n=t_n[1]
    t_loc_n = (c_loc_n * d_loc_n) / f_loc_n
    return t_loc_n

#  ∀v_n ∈ V_t, ∀v_m ∈ V_s_n
def t_direct(v_n,v_m,t_n,candidate_relay,vehicles):
    alpha_n_m=inrange(v_n,v_m)
    beta_n_m=isrelay(v_n,v_m,candidate_relay)
    d_off_n_m=t_n[0]
    r_n_m=get_transmission_rate(v_n,v_m,vehicles)
    c_off_m=t_n[1]
    f_off_m=v_m.computing_capacity

    t_direct_n_m=alpha_n_m*(1-beta_n_m)*((d_off_n_m/r_n_m)+((c_off_m*d_off_n_m)/f_off_m))
    return t_direct_n_m

def t_relay(v_n,v_m,v_prime_list,t_n,candidate_relay,vehicles):
    alpha_n_m=inrange(v_n,v_m)
    beta_n_m=isrelay(v_n,v_m,candidate_relay)
    d_off_n_m=t_n[0]
    r_n_m=get_transmission_rate(v_n,v_m,vehicles)

    latency_list=[]
    for v_p in v_prime_list:
        alpha_m_p=inrange(v_m,v_p)
        beta_m_p=isrelay(v_m,v_p,candidate_relay)
        r_m_p=get_transmission_rate(v_m,v_p,vehicles)
        c_off_p=t_n[1]
        f_off_p=v_p.computing_capacity

        t_relay_n_m_p=alpha_n_m*beta_n_m*((d_off_n_m/r_n_m) + alpha_m_p*(1-beta_m_p)*((d_off_n_m/r_m_p)+((c_off_p*d_off_n_m)/f_off_p)))
        latency_list.append(t_relay_n_m_p)

    return min(latency_list)

def off_latency(vehicles,vehicle_strategy, task_generated, Tavs,candidate_sn,candidate_relay,task_vehicle):
    sum=0
    last=vehicle_strategy[-1]
    d_off_n_m=task_generated[0]
    for v_m in vehicle_strategy :
        if(v_m.id!=last.id and v_m.id!=task_vehicle.id):
            sum+=inrange(task_vehicle,v_m)*(d_off_n_m/get_transmission_rate(task_vehicle,v_m,vehicles))
    
    prime_list=primelist(vehicles,Tavs,candidate_sn,candidate_relay,task_vehicle)

    t_off_n = sum + ((1-isrelay(task_vehicle,last,candidate_relay))*t_direct(task_vehicle,last,task_generated,candidate_relay,vehicles)) + isrelay(task_vehicle,last,candidate_relay)*t_relay(task_vehicle,last,prime_list,task_generated,candidate_relay,vehicles)
    return t_off_n
def primelist(vehicles,Tavs,candidate_sn,candidate_relay,task_vehicle):
    
    servicenodes_in_tavcluster=[]
    relaynodes_in_tavcluster=[]
   

    for i,j in zip(candidate_sn[task_vehicle.id],candidate_relay[task_vehicle.id]):
        
        servicenodes_in_tavcluster.append(vehicles[i[0].id-1])
        relaynodes_in_tavcluster.append(vehicles[j[0].id-1])
    service_relay_list=list(set(vehicles)-set(servicenodes_in_tavcluster+relaynodes_in_tavcluster+Tavs))
    return service_relay_list


def final_offloading_strategy(vehicle_strategy_list, task_vehicle, task_generated,vehicles,Tavs,candidate_sn,candidate_relay):
    final_strategy=vehicle_strategy_list[0]
    min_latency=float('inf')
    for vehicle_strategy in vehicle_strategy_list:
        tof=off_latency(vehicles,vehicle_strategy, task_generated,Tavs,candidate_sn,candidate_relay,task_vehicle)
        if(tof<min_latency):
            min_latency=tof
            final_strategy=vehicle_strategy
    return final_strategy

# if __name__ == "__main__":
#     # Example data
#     V={}
#     vehicles=vehicles_setup.main()
#     for i, vehicle in enumerate(vehicles, start=1):
#         print(f"Vehicle {i}: Id: {vehicle.id},Computation Cycles/s: {vehicle.computation_cycles_per_sec}, Transmission Power: {vehicle.transmission_power}, Radius: {vehicle.radius}, Position: {vehicle.get_curr_pos()}, Speed_Vector: {vehicle.speed_vec}")
#     dataset=[]
#     for i in vehicles:
#         dataset.append(i)
#     #dataset.append(len(vehicles)+1)
#     Tavs=[]
#     for i in range(5):
#         ele=random.randint(0,14)
#         while(dataset[ele] in Tavs):
#             ele=random.randint(0,14)
#         Tavs.append(vehicles[ele])
#     pbclustering=clustering.PartitionBasedClustering(dataset,Tavs)
#     candidate_sn, candidate_relay, sim_measure,relay_node=pbclustering.partition_based_clustering(0, 2)
#     dbclustering=clustering.DensityBasedClustering(candidate_sn, candidate_relay, sim_measure,relay_node)
#     candidate_sn,candidate_relay,relay_node=dbclustering.density_based_clustering(len(vehicles))
#     #dbclustering.print_cluster_density()
#     v0=clustering.find_path(candidate_sn, candidate_relay, relay_node, Tavs)
#     print("----------------------------------------------end----------------------------------------------------------------------------")

    
#     for i in Tavs:
#         print(final_offloading_strategy(v0[i.id], i, i.task,vehicles,Tavs,candidate_sn,candidate_relay))


    

    
    